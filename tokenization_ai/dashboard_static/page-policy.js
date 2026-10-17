import { el, clear, card, badge, button, field, icon } from "./ui.js";
import { verifyPolicyText } from "./api.js";

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
  const resultTone = statusTone(report.status);
  const result = el("div", "policy-result " + resultTone);
  result.appendChild(el("div", "policy-result-status", report.status));
  const resultText = report.solver_result && report.solver_result.is_entailment
    ? "Entailment established by the negated-conclusion query."
    : "The solver status is not, by itself, proof that the policy is valid.";
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
  solverBody.appendChild(el("div", "solver-line", "Query type: " + (solver.query_type || "not returned")));
  solverBody.appendChild(el("div", "solver-line", "Entailment established: " + (solver.is_entailment ? "Yes" : "No")));
  solverBody.appendChild(el("p", "card-note", solver.explanation || "No solver explanation was returned."));
  if (solver.model) { solverBody.appendChild(codeBlock(solver.model)); }
  stages.appendChild(stage("Z3 verification", "shield", solverBody, resultTone));

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

async function verify(text, policyId, output, submit) {
  submit.disabled = true;
  submit.textContent = "Verifying…";
  clear(output);
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
  form.appendChild(field("Natural-language policy", text));
  form.appendChild(field("Policy identifier", id));
  form.appendChild(submit);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!text.value.trim()) {
      renderError(output, "Enter a policy using the supported grammar.");
      return;
    }
    verify(text.value.trim(), id.value.trim(), output, submit);
  });
  formCard.body.appendChild(form);
  root.appendChild(formCard.root);

  const resultCard = card({ title: "Verification workspace", icon: "layers", badge: badge("Awaiting submission", "pending") });
  resultCard.body.appendChild(output);
  root.appendChild(resultCard.root);
}

export function unmount() {
  mounted = false;
}
