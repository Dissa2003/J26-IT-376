/* LogiShield Member 4 demo client.
 *
 * Presentation only: it posts a synthetic fixture to the same-origin
 * POST /v1/analyze endpoint and renders whatever the backend returns.
 * It performs no masking, tokenization, scoring or response generation,
 * and it never writes request or response data to the console or to
 * browser storage. All text is inserted with textContent.
 */
(function () {
  "use strict";

  var API_URL = "/v1/analyze";
  var SCENARIOS_URL = "/demo/assets/scenarios.json";
  var CONTRACT = "member4-analysis-v2";
  var TIMEOUT_MS = 15000;

  var scenarios = [];
  var selectedId = null;
  var renderedId = null;
  var busy = false;

  var $ = function (id) { return document.getElementById(id); };

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function clear(node) { while (node.firstChild) { node.removeChild(node.firstChild); } }

  function setStatus(kind, text) {
    var node = $("status");
    node.className = "status" + (kind ? " " + kind : "");
    node.textContent = text;
  }

  function kv(target, rows) {
    clear(target);
    rows.forEach(function (row) {
      target.appendChild(el("dt", null, row[0]));
      target.appendChild(el("dd", null, row[1]));
    });
  }

  function fmtMs(value) { return Number(value).toFixed(3); }

  /* ---- scenarios ---- */

  function currentScenario() {
    for (var i = 0; i < scenarios.length; i += 1) {
      if (scenarios[i].id === selectedId) { return scenarios[i]; }
    }
    return null;
  }

  function renderScenarios() {
    var list = $("scenario-list");
    clear(list);
    scenarios.forEach(function (scenario) {
      var button = el("button", "scenario");
      button.type = "button";
      button.setAttribute("role", "radio");
      button.setAttribute("aria-checked", String(scenario.id === selectedId));
      button.appendChild(el("strong", null, scenario.title));
      button.appendChild(el("span", null, scenario.description));
      button.addEventListener("click", function () { selectScenario(scenario.id); });
      list.appendChild(button);
    });
  }

  function selectScenario(id) {
    if (busy) { return; }
    if (renderedId !== null && renderedId !== id) {
      // Do not leave the previous scenario's result under the new request.
      $("stages").hidden = true;
      $("evidence").hidden = true;
      $("error-box").hidden = true;
      $("empty").hidden = false;
      renderedId = null;
    }
    selectedId = id;
    renderScenarios();
    var scenario = currentScenario();
    if (!scenario) { return; }
    $("request-view").textContent = JSON.stringify(scenario.request, null, 2);

    var source = scenario.request.rule_source;
    var box = $("rule-box");
    box.hidden = false;
    if (source === "mock_fixture") {
      $("rule-chip").textContent = "MOCK rule fixture";
      $("rule-text").textContent =
        "Illustrative mock data, not genuine Member 2 verification output. " +
        "This service does not verify rules.";
    } else {
      $("rule-chip").textContent = "No verified-rule input";
      $("rule-text").textContent =
        "No rules supplied. This is not a verification result.";
    }
    $("analyze-btn").disabled = false;
    setStatus("", "Ready. Press Analyze to send this synthetic request to the backend.");
  }

  /* ---- result rendering (backend data only) ---- */

  function showError(text) {
    var box = $("error-box");
    box.textContent = text;
    box.hidden = false;
    $("stages").hidden = true;
    $("evidence").hidden = true;
    $("empty").hidden = true;
  }

  function stageSummary(stages, name) {
    for (var i = 0; i < stages.length; i += 1) {
      if (stages[i].stage === name) { return stages[i]; }
    }
    return null;
  }

  function summaryText(summary) {
    var keys = Object.keys(summary || {});
    if (!keys.length) { return "none recorded"; }
    return keys.map(function (k) { return k + ": " + summary[k]; }).join(", ");
  }

  function render(data) {
    var stages = data.timings.stages;
    var rule = data.provenance.rule_status;
    var rep = data.masked_representation;

    kv($("st-context"), [
      ["Event id", data.event_id],
      ["Method", rep.method],
      ["Rule source", rule.label],
      ["Contract", data.contract_version]
    ]);

    var fusion = stageSummary(stages, "fusion");
    kv($("st-fusion"), [
      ["Rules fused", rule.rule_count],
      ["Rule ids shown", rule.rule_ids.length ? rule.rule_ids.join(", ") : "none"],
      ["Rule note", rule.note],
      ["Verified by this service", rule.verification_performed_by_this_service ? "yes" : "no"],
      ["Stage time", fusion ? fmtMs(fusion.elapsed_ms) + " ms" : "not recorded"]
    ]);

    var masking = stageSummary(stages, "masking");
    var norm = stageSummary(stages, "normalization");
    kv($("st-masking"), [
      ["Fields masked", data.masking.masked_field_count],
      ["Values hidden in preview", data.privacy.hidden_in_representation],
      ["Masking time", masking ? fmtMs(masking.elapsed_ms) + " ms" : "not recorded"],
      ["Normalization time", norm ? fmtMs(norm.elapsed_ms) + " ms" : "not recorded"]
    ]);
    $("st-representation").textContent = JSON.stringify(
      { method: rep.method, path: rep.path, headers: rep.headers, payload: rep.payload },
      null, 2
    );
    var fields = $("st-fields");
    clear(fields);
    if (!data.masking.masked_fields.length) {
      fields.appendChild(el("li", "muted small", "No fields were masked for this request."));
    }
    data.masking.masked_fields.forEach(function (name) {
      fields.appendChild(el("li", "chip role-masked", name));
    });

    var tok = stageSummary(stages, "tokenization");
    kv($("st-tokens"), [
      ["Tokens produced", data.tokens.total],
      ["Tokens displayed", data.tokens.returned + (data.tokens.truncated ? " (truncated)" : "")],
      ["Literals hidden in preview", data.privacy.hidden_in_tokens],
      ["Stage time", tok ? fmtMs(tok.elapsed_ms) + " ms" : "not recorded"]
    ]);
    var list = $("st-token-list");
    clear(list);
    data.tokens.items.forEach(function (item) {
      var chip = el("li", "chip role-" + item.role, item.token);
      chip.title = item.role;
      list.appendChild(chip);
    });

    var tensor = data.tensor;
    var ten = stageSummary(stages, "tensorization");
    kv($("st-tensor"), [
      ["Fixed length", tensor.max_length],
      ["Tokens before truncation", tensor.token_count_before_truncation],
      ["Tokens in tensor", tensor.tokens_in_tensor],
      ["Padded positions", tensor.padded_positions],
      ["Payload truncated", tensor.payload_truncated ? "yes" : "no"],
      ["Unknown-token count", tensor.unknown_token_count],
      ["Vocabulary size", tensor.vocabulary_size],
      ["Stage time", ten ? fmtMs(ten.elapsed_ms) + " ms" : "not recorded"]
    ]);

    var score = data.prediction.score;
    var inf = stageSummary(stages, "model_inference");
    $("score-num").textContent = score.toFixed(4);
    $("score-bar").value = Math.max(0, Math.min(1, score));
    $("score-bar").setAttribute("aria-label", "Anomaly score " + score.toFixed(4) + " out of 1");
    kv($("st-prediction"), [
      ["Anomaly score", score.toFixed(4) + " (0 to 1)"],
      ["Model version", data.prediction.model_version],
      ["Inference time", inf ? fmtMs(inf.elapsed_ms) + " ms" : "not recorded"],
      ["Total pipeline time", fmtMs(data.timings.total_elapsed_ms) + " ms"]
    ]);
    $("st-caveat").textContent =
      data.provenance.model.note +
      " This score is not a verdict, not a threshold decision and not detection accuracy.";

    var body = $("timing-table").tBodies[0];
    clear(body);
    stages.forEach(function (stage) {
      var row = body.insertRow();
      row.insertCell().textContent = stage.stage;
      var ms = row.insertCell();
      ms.className = "num-cell";
      ms.textContent = fmtMs(stage.elapsed_ms);
      row.insertCell().textContent = stage.success ? summaryText(stage.summary) : "failed";
    });

    kv($("provenance"), [
      ["Input origin", data.provenance.input_origin],
      ["Rule source", rule.source],
      ["Rule status", rule.label],
      ["Model version", data.provenance.model.model_version],
      ["Accuracy claim", data.provenance.detection_accuracy_claim]
    ]);
    kv($("privacy"), [
      ["Output kind", data.privacy.output_kind],
      ["Policy", data.privacy.policy],
      ["Note", data.privacy.note]
    ]);

    $("error-box").hidden = true;
    $("empty").hidden = true;
    $("stages").hidden = false;
    $("evidence").hidden = false;
  }

  /* ---- analyze ---- */

  function describeError(status, payload) {
    var code = payload && typeof payload.error === "string" ? payload.error : "error";
    var detail = payload && typeof payload.detail === "string" ? payload.detail : "";
    return "The backend rejected the request (HTTP " + status + ", " + code + ")." +
      (detail ? " " + detail : "");
  }

  function analyze() {
    if (busy) { return; }
    var scenario = currentScenario();
    if (!scenario) { return; }

    busy = true;
    $("analyze-btn").disabled = true;
    $("analyze-label").textContent = "Analyzing…";
    $("results").setAttribute("aria-busy", "true");
    $("error-box").hidden = true;
    setStatus("loading", "Sending the synthetic request to POST /v1/analyze…");

    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, TIMEOUT_MS);

    fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scenario.request),
      signal: controller.signal,
      credentials: "same-origin",
      cache: "no-store"
    }).then(function (response) {
      return response.json().then(
        function (payload) { return { ok: response.ok, status: response.status, payload: payload }; },
        function () { return { ok: false, status: response.status, payload: null }; }
      );
    }).then(function (result) {
      if (!result.ok) {
        showError(describeError(result.status, result.payload));
        setStatus("fail", "Analysis failed (HTTP " + result.status + ").");
        return;
      }
      if (!result.payload || result.payload.contract_version !== CONTRACT) {
        showError("Unexpected response contract; expected " + CONTRACT + ".");
        setStatus("fail", "Analysis failed: unexpected response contract.");
        return;
      }
      render(result.payload);
      renderedId = scenario.id;
      setStatus("ok", "Analysis complete. These values came from the Member 4 service.");
    }).catch(function (error) {
      var timedOut = error && error.name === "AbortError";
      showError(timedOut
        ? "The backend did not respond within " + (TIMEOUT_MS / 1000) + " seconds."
        : "Could not reach the Member 4 service. Is it running on this machine?");
      setStatus("fail", "Analysis failed: backend unreachable.");
    }).then(function () {
      clearTimeout(timer);
      busy = false;
      $("analyze-btn").disabled = false;
      $("analyze-label").textContent = "Analyze";
      $("results").setAttribute("aria-busy", "false");
    });
  }

  /* ---- boot ---- */

  function boot() {
    $("analyze-btn").addEventListener("click", analyze);
    fetch(SCENARIOS_URL, { cache: "no-store", credentials: "same-origin" })
      .then(function (response) {
        if (!response.ok) { throw new Error("scenarios"); }
        return response.json();
      })
      .then(function (data) {
        scenarios = data.scenarios || [];
        var wanted = new URLSearchParams(window.location.search).get("scenario");
        renderScenarios();
        var first = scenarios.length ? scenarios[0].id : null;
        var match = scenarios.filter(function (s) { return s.id === wanted; })[0];
        selectScenario(match ? match.id : first);
        if (match && new URLSearchParams(window.location.search).get("autorun") === "1") {
          analyze();
        }
      })
      .catch(function () {
        clear($("scenario-list"));
        $("scenario-list").appendChild(el("p", "muted", "Could not load the demo scenarios."));
        setStatus("fail", "Demo scenarios unavailable.");
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
