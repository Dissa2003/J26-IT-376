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

/* Collapsible section using native <details>; closed by default. */
function disclosure(label, content) {
  const box = el("details", "pv-disclosure");
  box.appendChild(el("summary", null, label));
  box.appendChild(content);
  return box;
}

/* ---- Step scaffolding ---------------------------------------------------- */

const STEPS = [
  { title: "Policy input", icon: "file" },
  { title: "Attribute extraction", icon: "layers" },
  { title: "Predicate logic", icon: "code" },
  { title: "Formal calculus check", icon: "merge" },
  { title: "Constraint encoding", icon: "lock" },
  { title: "Z3 solver result", icon: "shield" },
  { title: "Final decision", icon: "decisions" }
];

function createStep(index) {
  const spec = STEPS[index];
  const root = el("article", "pv-step is-pending");
  const head = el("div", "pv-step-head");
  head.appendChild(el("span", "pv-step-number", String(index + 1)));
  const marker = el("span", "policy-stage-marker");
  marker.appendChild(icon(spec.icon, 18));
  head.appendChild(marker);
  head.appendChild(el("h3", null, "Step " + (index + 1) + " — " + spec.title));
  const status = badge("Not reached", "pending");
  head.appendChild(status);
  const waiting = el("p", "pv-waiting", "Waiting for the previous step.");
  const body = el("div", "pv-step-body");
  body.hidden = true;
  root.appendChild(head);
  root.appendChild(waiting);
  root.appendChild(body);
  return { root, body, status, waiting };
}

function setStatus(step, text, tone) {
  step.status.textContent = text;
  step.status.className = "badge badge-" + tone;
}

function resetStep(step) {
  step.root.className = "pv-step is-pending";
  clear(step.body);
  step.body.hidden = true;
  step.waiting.hidden = false;
  setStatus(step, "Not reached", "pending");
}

function revealStep(step, content, text, tone) {
  clear(step.body);
  step.body.appendChild(content);
  step.body.hidden = false;
  step.waiting.hidden = true;
  step.root.className = "pv-step is-revealed";
  setStatus(step, text, tone);
}

function arrow() {
  const node = el("div", "pv-arrow");
  node.setAttribute("aria-hidden", "true");
  node.appendChild(icon("arrowDown", 20));
  return node;
}

/* ---- Step content builders ------------------------------------------------ */

function attributesView(attributes) {
  const grid = el("div", "pv-attr-grid");
  [["Role", attributes.subject], ["Action", attributes.action], ["Resource", attributes.resource]].forEach((pair) => {
    const box = el("div", "pv-attr");
    box.appendChild(el("span", "pv-attr-label", pair[0]));
    box.appendChild(el("span", "pv-attr-value", pair[1] || "Not returned"));
    grid.appendChild(box);
  });
  const wrap = el("div", "pv-stack");
  wrap.appendChild(grid);
  wrap.appendChild(el("p", "card-note", "Effect: " + (attributes.effect || "not returned")
    + " · Interpretation: " + (attributes.interpretation || "not returned")));
  return wrap;
}

/* The sentence follows the policy's meaning; the three forms are not equivalent. */
function sentenceParts(attributes) {
  const role = attributes.subject || "?";
  const action = attributes.action || "?";
  const resource = attributes.resource || "?";
  if (attributes.interpretation === "only_role") {
    return [["IF", "kw"], [" a user can ", null], [action, "val"], [" on ", null], [resource, "val"],
      [" THEN", "kw"], [" they must have role ", null], [role, "val"]];
  }
  if (attributes.effect === "Deny" || attributes.interpretation === "role_prohibition") {
    return [["IF", "kw"], [" a user has role ", null], [role, "val"], [" THEN", "kw"],
      [" they cannot ", null], [action, "val"], [" on ", null], [resource, "val"]];
  }
  return [["IF", "kw"], [" a user has role ", null], [role, "val"], [" THEN", "kw"],
    [" they can ", null], [action, "val"], [" on ", null], [resource, "val"]];
}

function predicateView(report) {
  const wrap = el("div", "pv-stack");
  const sentence = el("p", "pv-sentence");
  sentenceParts(report.attributes || {}).forEach((part) => {
    sentence.appendChild(el(part[1] === "kw" ? "span" : part[1] === "val" ? "strong" : "span",
      part[1] === "kw" ? "pv-kw" : null, part[0]));
  });
  wrap.appendChild(sentence);
  const formula = report.fol_formula && report.fol_formula.formula;
  wrap.appendChild(el("code", "pv-formula", formula || "Formula not returned."));
  if (report.structured_policy) {
    wrap.appendChild(disclosure("Show structured policy", codeBlock(report.structured_policy)));
  }
  return wrap;
}

function hasConflict(report) {
  return (Array.isArray(report.conflicts) && report.conflicts.length > 0)
    || report.rule_status === "conflicted" || report.rule_status === "contradictory";
}

function calculusView(report) {
  const wrap = el("div", "pv-stack");
  const conflicted = hasConflict(report);
  const banner = el("div", "pv-banner " + (conflicted ? "bad" : "live"));
  banner.appendChild(el("div", "pv-banner-title", conflicted
    ? "❌ Conflict detected — rules contradict each other"
    : "✅ No conflicts — rules are consistent"));
  banner.appendChild(el("p", null, conflicted
    ? "The conflict detector or consistency check rejected this policy."
    : "No permission/prohibition conflict was found for this policy. Several policies are checked against each other by POST /api/v1/verify-policies."));
  wrap.appendChild(banner);

  const inference = el("div", "policy-inference");
  if (Array.isArray(report.inference_trace) && report.inference_trace.length) {
    report.inference_trace.forEach((step) => {
      const row = el("div", "inference-step");
      row.appendChild(el("span", "step-number", String(step.step_number)));
      const body = el("div");
      body.appendChild(el("strong", null, step.rule));
      (step.premises || []).forEach((premise) => body.appendChild(el("code", "pv-formula", premise)));
      body.appendChild(el("code", "pv-formula pv-conclusion", "⊢ " + step.conclusion));
      body.appendChild(el("p", "card-note", step.explanation));
      row.appendChild(body);
      inference.appendChild(row);
    });
  } else {
    inference.appendChild(el("p", "empty-note", "No inference steps were returned."));
  }
  wrap.appendChild(inference);

  if (Array.isArray(report.conflicts) && report.conflicts.length) {
    const conflicts = el("div", "policy-conflicts");
    report.conflicts.forEach((conflict) => {
      conflicts.appendChild(el("p", null, conflict.explanation));
      conflicts.appendChild(codeBlock(conflict));
    });
    wrap.appendChild(disclosure("Show conflict details", conflicts));
  }
  return wrap;
}

function encodingView(report) {
  const wrap = el("div", "pv-stack");
  const constraints = Array.isArray(report.constraints) ? report.constraints : [];
  const rule = constraints.find((item) => item.kind === "rule");
  wrap.appendChild(el("p", "card-note", "The rule constraint passed to the solver, as readable first-order logic:"));
  wrap.appendChild(codeBlock(rule ? rule.expression : "No rule constraint was returned."));
  if (constraints.length) {
    const list = el("div", "pv-constraint-list");
    constraints.forEach((item) => {
      const row = el("div", "pv-constraint");
      row.appendChild(badge(item.kind, "info"));
      row.appendChild(el("code", "pv-formula", item.expression));
      list.appendChild(row);
    });
    wrap.appendChild(disclosure("Show full encoding (" + constraints.length + " constraints)", list));
  }
  return wrap;
}

function solverView(report) {
  const solver = report.solver_result || {};
  const status = report.status || "UNKNOWN";
  const isEntailed = status === "UNSAT" && solver.is_entailment === true;
  const wrap = el("div", "pv-stack");

  const row = el("div", "pv-solver-row");
  row.appendChild(el("span", "pv-solver-badge " + (isEntailed ? "live" : "mock"), status));
  const meaning = el("div");
  meaning.appendChild(el("div", "pv-banner-title", isEntailed
    ? "Verified – conclusion entailed"
    : status === "UNKNOWN" ? "Solver could not decide" : "Conclusion not entailed"));
  meaning.appendChild(el("p", "card-note", isEntailed
    ? "Raw solver status: " + status + ". For an entailment query UNSAT means the negated conclusion is impossible, so the proof succeeded."
    : "Raw solver status: " + status + ". The solver status is not, by itself, proof that the policy is valid."));
  row.appendChild(meaning);
  wrap.appendChild(row);

  wrap.appendChild(el("p", "card-note", "Demonstration check (synthetic request). The request here is generated from the policy itself. Use 'Check a request' below for an independent request."));
  if (solver.is_entailment === true) {
    const caveat = el("div", "caveat", "Note: This proof uses a fixed demonstration individual (Alice) derived from the policy itself. The entailment holds by construction. Use the 'Check a request' form below to evaluate real access decisions.");
    caveat.setAttribute("role", "note");
    wrap.appendChild(caveat);
  }

  const explanation = report.explanation || {};
  const details = el("div", "pv-stack");
  details.appendChild(el("div", "solver-line", "Query type: " + (solver.query_type || "not returned")));
  details.appendChild(el("div", "solver-line", "Entailment established: " + (solver.is_entailment ? "Yes" : "No")));
  details.appendChild(el("p", "card-note", solver.explanation || "No solver explanation was returned."));
  if (solver.model) { details.appendChild(codeBlock(solver.model)); }
  details.appendChild(el("h4", null, "Explanation"));
  details.appendChild(codeBlock(explanation));
  details.appendChild(el("h4", null, "Rule-cache eligibility"));
  details.appendChild(codeBlock({
    status: report.rule_status || "not returned",
    formally_established: explanation.formally_established === true,
    limitations: report.limitations || explanation.limitations || []
  }));
  wrap.appendChild(disclosure("Show solver details and rule-cache eligibility", details));
  return { content: wrap, isEntailed };
}

/* ---- Errors ---------------------------------------------------------------- */

function renderError(root, message, unsupported = false) {
  clear(root);
  const box = el("div", "policy-error " + (unsupported ? "policy-unsupported" : ""));
  box.setAttribute("role", "alert");
  box.appendChild(el("strong", null, unsupported ? "Unsupported policy" : "Verification request failed"));
  box.appendChild(el("p", null, message));
  root.appendChild(box);
}

/* ---- Step 7: independent request and final decision ----------------------- */

const REASON_WORDS = {
  request_entailed: "The request is allowed by the policies.",
  request_prohibited: "The request is denied by the policies.",
  not_covered_by_policies: "Policies do not decide this request.",
  policy_set_contradictory: "The submitted policies contradict each other.",
  solver_unknown: "The solver could not determine the request."
};

const DECISION_VIEW = {
  ALLOWED: ["✅ ALLOWED", "live"],
  DENIED: ["❌ DENIED", "bad"],
  NOT_DETERMINED: ["⚠️ NOT_DETERMINED", "mock"]
};

function queryRow(label, query) {
  const row = el("div", "request-query-row");
  row.appendChild(el("strong", null, label));
  row.appendChild(badge(query && query.status ? query.status.toUpperCase() : "UNKNOWN", statusTone(query && query.status ? query.status.toUpperCase() : "UNKNOWN")));
  return row;
}

function bulletList(items) {
  const list = el("ul", "pv-list");
  (items || []).forEach((item) => list.appendChild(el("li", null, item)));
  if (!list.firstChild) { list.appendChild(el("li", null, "None returned.")); }
  return list;
}

function requestReport(root, report) {
  clear(root);
  const decision = DECISION_VIEW[report.decision] ? report.decision : "NOT_DETERMINED";
  const view = DECISION_VIEW[decision];
  const result = el("div", "pv-decision " + view[1]);
  result.setAttribute("role", "status");
  result.appendChild(el("div", "pv-decision-value", view[0]));
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
  details.appendChild(bulletList(report.assumptions));
  details.appendChild(el("h3", null, "Limitations"));
  details.appendChild(bulletList(report.limitations));
  root.appendChild(disclosure("Show Z3 queries, assumptions and limitations", details));
  return decision;
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

function createRequestPanel(report, step) {
  const wrap = el("div", "pv-stack");
  wrap.appendChild(el("h4", "pv-subhead", "Check a request"));
  wrap.appendChild(el("p", "card-note", "Enter an independent subject and role assignment. The decision comes from POST /api/v1/verify-request."));
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
  output.appendChild(el("p", "pv-waiting", "No request checked yet — the final decision appears here."));
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
        setStatus(step, "Request failed", "bad");
        return;
      }
      const decision = requestReport(output, result.payload);
      setStatus(step, decision, DECISION_VIEW[decision][1]);
    } catch (error) {
      requestError(output, "The request verification API could not be reached. Try again.");
      setStatus(step, "Request failed", "bad");
    } finally {
      submit.disabled = false;
      submit.textContent = "Check request";
    }
  });
  wrap.appendChild(form);
  wrap.appendChild(output);
  return wrap;
}

/* ---- Pipeline run ------------------------------------------------------------ */

async function verify(text, policyId, steps, inputOutput, submit, workspaceBadge) {
  submit.disabled = true;
  submit.textContent = "Verifying…";
  workspaceBadge.textContent = "AWAITING SUBMISSION";
  workspaceBadge.className = "badge badge-pending";
  steps.slice(1).forEach(resetStep);
  clear(inputOutput);
  inputOutput.appendChild(el("p", "loading-note", "Running parser, calculus, constraints and Z3 verification…"));
  setStatus(steps[0], "Verifying…", "info");
  try {
    const result = await verifyPolicyText(text, policyId);
    const payload = result.payload;
    if (!result.ok) {
      const detail = payload && payload.detail;
      const message = detail && typeof detail.message === "string"
        ? detail.message
        : "The backend returned HTTP " + result.status + ".";
      const unsupported = detail && detail.code === "unsupported_policy";
      renderError(inputOutput, message, unsupported);
      setStatus(steps[0], unsupported ? "Unsupported" : "Failed", "bad");
      return;
    }
    clear(inputOutput);
    inputOutput.appendChild(el("p", "card-note", "Submitted: " + (payload.original_policy || text)));
    setStatus(steps[0], "Submitted", "live");
    revealStep(steps[1], attributesView(payload.attributes || {}), "Extracted", "live");
    revealStep(steps[2], predicateView(payload), "Translated", "live");
    const conflicted = hasConflict(payload);
    revealStep(steps[3], calculusView(payload), conflicted ? "Conflict" : "Consistent", conflicted ? "bad" : "live");
    revealStep(steps[4], encodingView(payload), "Encoded", "live");
    const solver = solverView(payload);
    revealStep(steps[5], solver.content, payload.status || "UNKNOWN", solver.isEntailed ? "live" : "mock");
    revealStep(steps[6], createRequestPanel(payload, steps[6]), "Awaiting request", "pending");
    workspaceBadge.textContent = "COMPLETE";
    workspaceBadge.className = "badge badge-live";
  } catch (error) {
    renderError(inputOutput, "The verification API could not be reached. Start the local FastAPI service and try again.");
    setStatus(steps[0], "Failed", "bad");
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

  const workspaceBadge = badge("AWAITING SUBMISSION", "pending");
  const flowCard = card({ title: "Verification workspace", icon: "layers", badge: workspaceBadge });
  const flow = el("div", "pv-flow");
  const steps = STEPS.map((_, index) => createStep(index));

  const form = el("form", "policy-form");
  const text = el("textarea", "policy-input");
  text.rows = 3;
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
  const inputOutput = el("div", "policy-output");
  form.appendChild(field("Enter a plain-English access policy", text));
  form.appendChild(el("p", "card-note", "Supported forms: '<role> can <action> <resource>', 'only <role> can …', '<role> cannot …'."));
  form.appendChild(field("Policy identifier", id));
  form.appendChild(submit);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!text.value.trim()) {
      renderError(inputOutput, "Enter a policy using the supported grammar.");
      return;
    }
    verify(text.value.trim(), id.value.trim(), steps, inputOutput, submit, workspaceBadge);
  });
  const inputBody = el("div", "pv-stack");
  inputBody.appendChild(form);
  inputBody.appendChild(inputOutput);
  revealStep(steps[0], inputBody, "Ready", "info");

  steps.forEach((step, index) => {
    if (index > 0) { flow.appendChild(arrow()); }
    flow.appendChild(step.root);
  });
  flowCard.body.appendChild(flow);
  root.appendChild(flowCard.root);
}

export function unmount() {
  mounted = false;
}
