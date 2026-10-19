import { el, clear, card, badge, button, field, icon } from "./ui.js";
import { verifyPolicyText, verifyRequest } from "./api.js";

const EXAMPLE = "Only administrators can delete user accounts.";
let mounted = false;

function codeBlock(value) {
  const pre = el("pre", "policy-code");
  pre.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return pre;
}

function statusTone(status) {
  if (status === "SAT") { return "live"; }
  if (status === "UNSAT") { return "bad"; }
  if (status === "UNKNOWN") { return "mock"; }
  return "pending";
}

function stage(title, iconName, content, tone = "info") {
  const item = el("article", "policy-stage");
  const head = el("div", "policy-stage-head");
  const marker = el("span", "policy-stage-marker");
  marker.appendChild(icon(iconName, 18));
  head.appendChild(marker);
  head.appendChild(el("h3", null, title));
  head.appendChild(badge(tone === "live" ? "Complete" : "Returned", tone));
  item.appendChild(head);
  item.appendChild(content);
  return item;
}

function renderReport(root, report) {
  clear(root);
  const hasConflict = Array.isArray(report.conflicts) && report.conflicts.length > 0;
  const isEntailed = report.status === "UNSAT"
    && report.solver_result
    && report.solver_result.is_entailment === true;
  const resultTone = hasConflict ? "bad" : isEntailed ? "live" : "pending";
  const result = el("div", "policy-result " + resultTone);
  result.appendChild(el("div", "policy-result-status", isEntailed
    ? "Verified – conclusion entailed"
    : hasConflict ? "Policy conflict detected" : "Verification returned"));
  const resultText = isEntailed
    ? "Raw solver status: " + report.status
    : report.solver_result && report.solver_result.is_entailment
      ? "Raw solver status: " + report.status + ". Entailment established by the negated-conclusion query."
      : "Raw solver status: " + report.status + ". The solver status is not, by itself, proof that the policy is valid.";
  result.appendChild(el("p", null, resultText));
  root.appendChild(result);

  const stages = el("div", "policy-stages");
  stages.appendChild(stage("Policy input", "file", codeBlock(report.original_policy || "Not returned.")));
  stages.appendChild(stage("Attribute extraction", "layers", codeBlock(report.attributes || "Not returned.")));
  stages.appendChild(stage("Structured representation", "grid", codeBlock(report.structured_policy || "Not returned.")));
  stages.appendChild(stage("Predicate logic", "code", codeBlock(report.fol_formula || "Not returned.")));

  const inference = el("div", "policy-inference");
  if (Array.isArray(report.inference_trace) && report.inference_trace.length) {
    report.inference_trace.forEach((step) => {
      const row = el("div", "inference-step");
      row.appendChild(el("span", "step-number", String(step.step_number)));
      const body = el("div");
      body.appendChild(el("strong", null, step.rule));
      body.appendChild(codeBlock({
        premises: step.premises,
        conclusion: step.conclusion,
        explanation: step.explanation
      }));
      row.appendChild(body);
      inference.appendChild(row);
    });
  } else {
    inference.appendChild(el("p", "empty-note", "No inference steps were returned."));
  }
  stages.appendChild(stage("Formal calculus", "merge", inference));
  stages.appendChild(stage("Constraint generation", "lock", codeBlock(report.constraints || "Not returned.")));

  const solver = report.solver_result || {};
  const solverBody = el("div", "solver-summary");
  solverBody.appendChild(el("p", "card-note", "The request here is generated from the policy itself. Use 'Check a request' below for an independent request."));
  solverBody.appendChild(el("div", "solver-line", "Query type: " + (solver.query_type || "not returned")));
  solverBody.appendChild(el("div", "solver-line", "Entailment established: " + (solver.is_entailment ? "Yes" : "No")));
  solverBody.appendChild(el("p", "card-note", solver.explanation || "No solver explanation was returned."));
  if (solver.model) { solverBody.appendChild(codeBlock(solver.model)); }
  stages.appendChild(stage("Demonstration check (synthetic request)", "shield", solverBody, resultTone));
  if (solver.is_entailment === true) {
    const caveat = el("div", "caveat", "Note: This proof uses a fixed demonstration individual (Alice) derived from the policy itself. The entailment holds by construction. Use the 'Check a request' form below to evaluate real access decisions.");
    caveat.setAttribute("role", "note");
    stages.appendChild(caveat);
  }

  const explanation = report.explanation || {};
  stages.appendChild(stage("Explanation", "info", codeBlock(explanation), "info"));
  stages.appendChild(stage("Rule-cache eligibility", "database", codeBlock({
    status: report.rule_status || "not returned",
    formally_established: explanation.formally_established === true,
    limitations: report.limitations || explanation.limitations || []
  }), report.rule_status === "eligible" ? "live" : "pending"));

  if (Array.isArray(report.conflicts) && report.conflicts.length) {
    const conflicts = el("div", "policy-conflicts");
    report.conflicts.forEach((conflict) => {
      conflicts.appendChild(el("p", null, conflict.explanation));
      conflicts.appendChild(codeBlock(conflict));
    });
    stages.appendChild(stage("Conflicts", "alert", conflicts, "bad"));
  }
  root.appendChild(stages);
}

function renderError(root, message, unsupported = false) {
  clear(root);
  const box = el("div", "policy-error " + (unsupported ? "policy-unsupported" : ""));
  box.setAttribute("role", "alert");
  box.appendChild(el("strong", null, unsupported ? "Unsupported policy" : "Verification request failed"));
  box.appendChild(el("p", null, message));
  root.appendChild(box);
}

const REASON_WORDS = {
  request_entailed: "The request is allowed by the policies.",
  request_prohibited: "The request is denied by the policies.",
  not_covered_by_policies: "Policies do not decide this request.",
  policy_set_contradictory: "The submitted policies contradict each other.",
  solver_unknown: "The solver could not determine the request."
};

function queryRow(label, query) {
  const row = el("div", "request-query-row");
  row.appendChild(el("strong", null, label));
  row.appendChild(badge(query && query.status ? query.status.toUpperCase() : "UNKNOWN", statusTone(query && query.status ? query.status.toUpperCase() : "UNKNOWN")));
  return row;
}

function requestReport(root, report) {
  clear(root);
  const decision = report.decision || "NOT_DETERMINED";
  const isUndetermined = decision === "NOT_DETERMINED";
  const result = el("div", "policy-result request-decision " + (
    decision === "ALLOWED" ? "live" : decision === "DENIED" ? "bad" : "mock"
  ));
  result.appendChild(el("div", "policy-result-status", isUndetermined
    ? "Policies do not decide this request"
    : decision));
  result.appendChild(el("p", null, REASON_WORDS[report.reason_code] || "The backend returned an unrecognised reason."));
  root.appendChild(result);

  const queries = el("div", "request-query-list");
  queries.appendChild(queryRow("Consistency query", report.consistency_query));
  queries.appendChild(queryRow("Allow query", report.allow_query));
  queries.appendChild(queryRow("Deny query", report.deny_query));
  const details = el("div", "request-details");
  details.appendChild(el("h3", null, "Z3 queries"));
  details.appendChild(queries);
  details.appendChild(el("h3", null, "Assumptions"));
  details.appendChild(codeBlock(report.assumptions || []));
  details.appendChild(el("h3", null, "Limitations"));
  details.appendChild(codeBlock(report.limitations || []));
  root.appendChild(details);
}

function requestError(root, message) {
  clear(root);
  const box = el("div", "policy-error");
  box.setAttribute("role", "alert");
  box.appendChild(el("strong", null, "Request check failed"));
  box.appendChild(el("p", null, message));
  root.appendChild(box);
}

function policyForRequest(report) {
  const attributes = report.attributes || {};
  return {
    policy_id: report.policy_id || "verified-policy",
    subject: attributes.subject || "",
    action: attributes.action || "",
    resource: attributes.resource || "",
    effect: attributes.effect || "Allow",
    raw_text: report.original_policy || "",
    interpretation: attributes.interpretation || "role_permission"
  };
}

function createRequestPanel(report) {
  const requestCard = card({
    title: "Check a request",
    icon: "shield",
    badge: badge("Independent request", "info")
  });
  const form = el("form", "policy-form request-form");
  const subject = el("input", "text-input");
  subject.type = "text";
  subject.required = true;
  subject.maxLength = 64;
  subject.value = "subject-1";
  const roles = el("input", "text-input");
  roles.type = "text";
  roles.maxLength = 512;
  roles.placeholder = "Administrators, Auditors";
  const attributes = report.attributes || {};
  const action = el("input", "text-input");
  action.type = "text";
  action.required = true;
  action.maxLength = 64;
  action.value = attributes.action || "";
  const resource = el("input", "text-input");
  resource.type = "text";
  resource.required = true;
  resource.maxLength = 128;
  resource.value = attributes.resource || "";
  const submit = button("Check request", "btn-primary", null, "play");
  submit.type = "submit";
  const output = el("div", "policy-output");
  form.appendChild(field("Subject ID", subject));
  form.appendChild(field("Roles (comma-separated)", roles));
  form.appendChild(field("Action", action));
  form.appendChild(field("Resource", resource));
  form.appendChild(submit);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!subject.value.trim() || !action.value.trim() || !resource.value.trim()) {
      requestError(output, "Subject ID, action and resource are required.");
      return;
    }
    submit.disabled = true;
    submit.textContent = "Checking…";
    clear(output);
    output.appendChild(el("p", "loading-note", "Checking the request against the verified policy…"));
    try {
      const result = await verifyRequest({
        policies: [policyForRequest(report)],
        request: {
          subject_id: subject.value.trim(),
          roles: roles.value.split(",").map((role) => role.trim()).filter(Boolean),
          action: action.value.trim(),
          resource: resource.value.trim()
        }
      });
      if (!result.ok) {
        const detail = result.payload && result.payload.detail;
        const message = result.networkError
          ? "The request verification API could not be reached. Start the local FastAPI service and try again."
          : detail && typeof detail.message === "string"
            ? detail.message
            : "The backend rejected the request (HTTP " + result.status + ").";
        requestError(output, message);
        return;
      }
      requestReport(output, result.payload);
    } catch (error) {
      requestError(output, "The request verification API could not be reached. Try again.");
    } finally {
      submit.disabled = false;
      submit.textContent = "Check request";
    }
  });
  requestCard.body.appendChild(form);
  requestCard.body.appendChild(output);
  return requestCard.root;
}

async function verify(text, policyId, output, submit, requestHost, workspaceBadge) {
  submit.disabled = true;
  submit.textContent = "Verifying…";
  workspaceBadge.textContent = "AWAITING SUBMISSION";
  workspaceBadge.className = "badge badge-pending";
  clear(output);
  clear(requestHost);
  requestHost.hidden = true;
  output.appendChild(el("p", "loading-note", "Running parser, calculus, constraints and Z3 verification…"));
  try {
    const result = await verifyPolicyText(text, policyId);
    const payload = result.payload;
    if (!result.ok) {
      const detail = payload && payload.detail;
      const message = detail && typeof detail.message === "string"
        ? detail.message
        : "The backend returned HTTP " + result.status + ".";
      renderError(output, message, detail && detail.code === "unsupported_policy");
      return;
    }
    renderReport(output, payload);
    workspaceBadge.textContent = "COMPLETE";
    workspaceBadge.className = "badge badge-live";
    clear(requestHost);
    requestHost.appendChild(createRequestPanel(payload));
    requestHost.hidden = false;
  } catch (error) {
    renderError(output, "The verification API could not be reached. Start the local FastAPI service and try again.");
  } finally {
    submit.disabled = false;
    submit.textContent = "Verify policy";
  }
}

export function mount(root, route) {
  mounted = true;
  const head = el("div", "page-head");
  head.appendChild(el("h1", "page-title", route.title));
  head.appendChild(el("p", "page-sub", route.subtitle));
  root.appendChild(head);

  const notice = el("div", "info-banner policy-live-note");
  notice.appendChild(badge("Live API", "live"));
  notice.appendChild(el("p", null, "Results below come from POST /api/v1/verify-text. SAT, UNSAT and UNKNOWN are solver statuses, not interchangeable verification conclusions."));
  root.appendChild(notice);

  const formCard = card({ title: "Submit a security policy", icon: "shield", badge: badge("Deterministic grammar", "info") });
  const form = el("form", "policy-form");
  const text = el("textarea", "policy-input");
  text.rows = 4;
  text.required = true;
  text.maxLength = 500;
  text.placeholder = EXAMPLE;
  text.value = EXAMPLE;
  const id = el("input", "text-input");
  id.type = "text";
  id.maxLength = 64;
  id.placeholder = "Optional policy ID";
  const submit = button("Verify policy", "btn-primary", null, "play");
  submit.type = "submit";
  const output = el("div", "policy-output");
  const workspaceBadge = badge("AWAITING SUBMISSION", "pending");
  form.appendChild(field("Natural-language policy", text));
  form.appendChild(field("Policy identifier", id));
  form.appendChild(submit);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!text.value.trim()) {
      renderError(output, "Enter a policy using the supported grammar.");
      return;
    }
    verify(text.value.trim(), id.value.trim(), output, submit, requestHost, workspaceBadge);
  });
  formCard.body.appendChild(form);
  root.appendChild(formCard.root);

  const resultCard = card({ title: "Verification workspace", icon: "layers", badge: workspaceBadge });
  resultCard.body.appendChild(output);
  root.appendChild(resultCard.root);
  const requestHost = el("div");
  requestHost.hidden = true;
  root.appendChild(requestHost);
}

export function unmount() {
  mounted = false;
}
