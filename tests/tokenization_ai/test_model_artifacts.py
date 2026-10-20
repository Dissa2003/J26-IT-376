"""Tests for model-artifact-1 (draft protocol heldout-eval-1.2.0-draft, section 12.12.5).

Models are small hand-built fixtures: ``LightweightAnomalyModel`` objects with explicitly assigned float64 weights and a bias (no ``fit``, no data, no research
metric). Artifacts live in pytest temporary directories outside the repository. No confirmatory seed is used and no confirmatory data exists.
"""

from __future__ import annotations

import ast
import base64
import copy
import json
import math
import os
import stat
from pathlib import Path

import numpy as np
import pytest

import tokenization_ai.heldout_eval.manifest as mf
import tokenization_ai.heldout_eval.model_artifacts as ma
import tokenization_ai.heldout_eval.numerics as numerics
import tokenization_ai.heldout_eval.runner as rn
from tokenization_ai.heldout_eval.arms import ARM_A, ARM_B, ARM_C, ARMS
from tokenization_ai.model import LightweightAnomalyModel
from tokenization_ai.models import ModelInput

REPO_ROOT = Path(ma.__file__).resolve().parents[2]
MANIFEST = "c" * 64
L = 29.5
LAMBDA = 118.0


def model_with(nb: int, seed: int = 0, scale: float = 0.05, bias: float = 0.125) -> LightweightAnomalyModel:
    rng = np.random.default_rng(seed)
    model = LightweightAnomalyModel(num_buckets=nb)
    model.weights = rng.normal(0.0, scale, nb)
    model.bias = bias
    model.is_trained = True  # explicit fixture state, no fit
    return model


def training(arm: str = ARM_A, **over) -> dict:
    base = {"train_ids_sha256": "a" * 64, "vocabulary_sha256": None if arm == ARM_C else "b" * 64, "max_length": None if arm == ARM_C else 192,
            "lambda_method": "lambda-power-1", "lambda_max": LAMBDA, "L": L}
    base.update(over)
    return base


def rate(c: float) -> float:
    return numerics.relative_learning_rate(c, L)


def write(directory, model, arm=ARM_A, nb=1024, ep=150, c=0.3, **over):
    kwargs = dict(repo_root=REPO_ROOT, arm=arm, num_buckets=nb, epochs=ep, relative_step_c=c, learning_rate=rate(c), manifest_sha256=MANIFEST,
                  training=training(arm))
    kwargs.update(over)
    return ma.write_model_artifact(directory, model, **kwargs)


def load(directory, written, arm=ARM_A, nb=1024, ep=150, c=0.3, **over):
    cid = rn.canonical_candidate_id(arm, nb, ep, c)
    kwargs = dict(expected_sha256=written.sha256, expected_arm=arm, expected_candidate_id=cid, expected_manifest_sha256=MANIFEST,
                  expected_learning_rate=rate(c))
    kwargs.update(over)
    return ma.load_model_artifact(directory, **kwargs)


def rewrite(written, fn):
    """Edit an artifact's JSON, write it back with a *recomputed* digest (a forgery that matches its own digest) and return the new digest."""
    path = Path(written.path)
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    data = json.loads(path.read_bytes())
    fn(data)
    new = mf.canonical_bytes(data)
    path.write_bytes(new)
    return mf.sha256_hex(new)


@pytest.fixture()
def out(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir()
    return d


def inputs():
    return [ModelInput(event_id="a", input_ids=[3, 7, 9, 0, 0], attention_mask=[1, 1, 1, 0, 0]),
            ModelInput(event_id="b", input_ids=[11, 2, 3, 5], attention_mask=[1, 1, 1, 1]),
            ModelInput(event_id="c", input_ids=[2_000_000_000, 5], attention_mask=[1, 1]),
            ModelInput(event_id="d", input_ids=[4], attention_mask=[1])]


# -- serialization and exactness -------------------------------------------------------------------------------------------------------------


def test_constants_and_schema():
    assert ma.ARTIFACT_VERSION == "model-artifact-1" and ma.ARTIFACT_KIND == "fitted_candidate_model"
    art = ma.build_artifact(model_with(1024), arm=ARM_A, num_buckets=1024, epochs=150, relative_step_c=0.3, learning_rate=rate(0.3),
                            manifest_sha256=MANIFEST, training=training())
    assert set(art) == ma._TOP_KEYS and art["authorization"]["asserted"] is False
    assert art["candidate"]["candidate_id"] == rn.canonical_candidate_id(ARM_A, 1024, 150, 0.3)
    assert art["candidate"]["learning_rate"] == rate(0.3) and art["candidate"]["learning_rate_hex"] == rate(0.3).hex()
    assert art["model_source_sha256"] == ma.model_source_sha256() and art["manifest_sha256"] == MANIFEST
    assert art["input_representation"] == {"arms_version": "arms-1", "arm": ARM_A, "model_input": "tensorized_token_ids"}
    assert ma.input_representation(ARM_C)["model_input"] == "arm_c_hashed_char_trigram_ids"
    assert art["model"]["fingerprint"]["weights_sha256"] == art["model"]["weights_sha256"] and art["model"]["is_trained"] is True


def test_serialization_is_deterministic_canonical_ascii(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    wa, wb = write(a, model_with(1024, seed=5)), write(b, model_with(1024, seed=5))
    bytes_a, bytes_b = Path(wa.path).read_bytes(), Path(wb.path).read_bytes()
    assert bytes_a == bytes_b and wa.sha256 == wb.sha256 == mf.sha256_hex(bytes_a) and wa.size == len(bytes_a)
    assert bytes_a.endswith(b"\n") and bytes_a.isascii() and mf.canonical_bytes(json.loads(bytes_a)) == bytes_a
    other = tmp_path / "c"
    other.mkdir()
    assert write(other, model_with(1024, seed=6)).sha256 != wa.sha256  # different parameters, different bytes


def test_float64_weights_and_bias_round_trip_bit_exactly(out):
    model = model_with(1024, seed=1)
    model.weights[:8] = [5e-324, -5e-324, -0.0, 0.0, 1.7976931348623157e308, -1.7976931348623157e308, 0.1 + 0.2, np.nextafter(1.0, 2.0)]
    model.bias = 0.1 + 0.2
    written = write(out, model)
    loaded = load(out, written)
    assert loaded.model.weights.tobytes() == model.weights.tobytes()  # every bit, including the sign of zero
    assert loaded.model.weights.dtype == np.float64 and loaded.model.bias == model.bias and loaded.model.bias.hex() == model.bias.hex()
    assert loaded.model.weights is not model.weights and loaded.model.weights.flags.writeable
    assert loaded.fingerprint == ma.sr.model_artifact_fingerprint(model)


@pytest.mark.parametrize("nb", [1024, 2048, 4096])
def test_every_approved_bucket_size_round_trips(out, nb):
    model = model_with(nb, seed=nb)
    written = write(out, model, nb=nb)
    loaded = load(out, written, nb=nb)
    assert loaded.model.num_buckets == nb and np.array_equal(loaded.model.weights, model.weights)
    assert written.size < ma.MAX_ARTIFACT_BYTES


def test_reconstructed_model_behaves_like_the_original(out):
    model = model_with(1024, seed=2)
    loaded = load(out, write(out, model)).model
    assert loaded.is_trained is True and loaded.num_buckets == 1024 and loaded.model_version == model.model_version == "ngram-linear-v1-trained"
    for item in inputs():
        assert loaded.predict(item) == model.predict(item)
        assert loaded._features(item.input_ids, item.attention_mask).tobytes() == model._features(item.input_ids, item.attention_mask).tobytes()
    assert numerics.raw_logits(loaded, inputs()) == numerics.raw_logits(model, inputs())
    assert numerics.verify_logit_compatibility(loaded, inputs())["checked"] == 4


def test_saturated_logits_are_preserved_exactly(out):
    big = model_with(1024, scale=0.0, bias=0.0)
    big.weights[:] = 400.0
    low = model_with(2048, scale=0.0, bias=-5000.0)
    low.weights[:] = 400.0
    loaded_big = load(out, write(out, big), arm=ARM_A, nb=1024).model
    loaded_low = load(out, write(out, low, arm=ARM_B, nb=2048, ep=50, c=0.8), arm=ARM_B, nb=2048, ep=50, c=0.8).model
    for original, loaded in ((big, loaded_big), (low, loaded_low)):
        assert numerics.raw_logits(loaded, inputs()) == numerics.raw_logits(original, inputs())
        assert [loaded.predict(i) for i in inputs()] == [original.predict(i) for i in inputs()]
    assert loaded_big.predict(inputs()[0]) == 1.0 and loaded_low.predict(inputs()[0]) == 0.0
    assert all(math.isfinite(x) for x in numerics.raw_logits(loaded_big, inputs()))


def test_the_loaded_model_is_independent_of_the_original_and_of_other_loads(out):
    model = model_with(1024, seed=3)
    written = write(out, model)
    first, second = load(out, written).model, load(out, written).model
    first.weights[0] += 1.0
    assert second.weights[0] == model.weights[0] and first.weights[0] != model.weights[0]


# -- write-once storage ----------------------------------------------------------------------------------------------------------------------


def test_file_name_comes_from_trusted_identity_only(out):
    written = write(out, model_with(1024))
    cid = rn.canonical_candidate_id(ARM_A, 1024, 150, 0.3)
    assert Path(written.path).name == ma.artifact_filename(MANIFEST, cid) and Path(written.path).parent == out
    assert Path(written.path).name.startswith("model-artifact-") and Path(written.path).name.endswith(".json") and len(Path(written.path).name) == 84
    for bad in ("../../escape", "A_request_only|nb=00001|ep=0001|c=0.10", "", None, "x" * 100, "..\\x"):
        with pytest.raises(ma.ArtifactSchemaError):
            ma.artifact_filename(MANIFEST, bad)  # type: ignore[arg-type]
    for bad in ("short", "../" + "c" * 61, None):
        with pytest.raises(ma.ArtifactSchemaError):
            ma.artifact_filename(bad, cid)  # type: ignore[arg-type]
    assert [p.name for p in out.iterdir()] == [Path(written.path).name]


def test_an_existing_artifact_is_never_overwritten(out):
    first = write(out, model_with(1024, seed=1))
    before = Path(first.path).read_bytes()
    with pytest.raises(ma.ArtifactWriteError):
        write(out, model_with(1024, seed=2))
    with pytest.raises(ma.ArtifactWriteError):
        write(out, model_with(1024, seed=1))
    assert Path(first.path).read_bytes() == before
    assert not (Path(first.path).stat().st_mode & stat.S_IWRITE)


def test_directory_restrictions(out, tmp_path):
    with pytest.raises(ma.ArtifactWriteError):
        write(REPO_ROOT / "tokenization_ai", model_with(1024))
    with pytest.raises(ma.ArtifactWriteError):
        write(REPO_ROOT, model_with(1024))
    with pytest.raises(ma.ArtifactWriteError):
        write(tmp_path / "missing", model_with(1024))
    assert not list((REPO_ROOT / "tokenization_ai").glob("model-artifact-*"))


def test_a_failed_write_is_left_in_place_and_blocks_a_retry(out, monkeypatch):
    def broken(fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", broken)
    with pytest.raises(ma.ArtifactWriteError):
        write(out, model_with(1024))
    monkeypatch.undo()
    assert len(list(out.iterdir())) == 1
    with pytest.raises(ma.ArtifactWriteError):
        write(out, model_with(1024))


def test_a_file_appearing_between_check_and_write_is_not_replaced(out, monkeypatch):
    target = out / ma.artifact_filename(MANIFEST, rn.canonical_candidate_id(ARM_A, 1024, 150, 0.3))
    real = os.path.lexists

    def late(path):
        if Path(path) == target:
            target.write_bytes(b"someone else's file")
            return False
        return real(path)

    monkeypatch.setattr(os.path, "lexists", late)
    with pytest.raises(ma.ArtifactWriteError):
        write(out, model_with(1024))
    assert target.read_bytes() == b"someone else's file"


def test_a_read_back_mismatch_leaves_the_file_for_review(out, monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"different")
    with pytest.raises(ma.ArtifactWriteError):
        write(out, model_with(1024))
    monkeypatch.undo()
    assert len(list(out.iterdir())) == 1


# -- invalid models, candidates, provenance (nothing is written) -----------------------------------------------------------------------------------------------------


def nothing_written(out):
    return list(out.iterdir()) == []


def test_invalid_model_states_are_rejected(out):
    nan_model = model_with(1024)
    nan_model.weights[5] = np.nan
    inf_model = model_with(1024)
    inf_model.weights[5] = np.inf
    nan_bias = model_with(1024)
    nan_bias.bias = float("nan")
    inf_bias = model_with(1024)
    inf_bias.bias = float("inf")
    short = model_with(1024)
    short.weights = np.zeros(7)
    f32 = model_with(1024)
    f32.weights = np.zeros(1024, dtype=np.float32)
    two_d = model_with(1024)
    two_d.weights = np.zeros((1024, 1))
    bool_bias = model_with(1024)
    bool_bias.bias = True
    for bad in (LightweightAnomalyModel(num_buckets=1024), nan_model, inf_model, nan_bias, inf_bias, short, f32, two_d, bool_bias, object(), None):
        with pytest.raises(ma.ArtifactError):
            write(out, bad)  # type: ignore[arg-type]
    assert nothing_written(out)


def test_model_and_candidate_bucket_counts_must_agree(out):
    with pytest.raises(ma.ArtifactVerificationError):
        write(out, model_with(2048), nb=1024)
    assert nothing_written(out)


@pytest.mark.parametrize("over", [dict(arm="D_other"), dict(nb=999), dict(nb=True), dict(ep=7), dict(ep=True), dict(c=0.5), dict(c=True), dict(c="0.3"),
                                  dict(arm=None)])
def test_unapproved_candidate_parameters_are_rejected(out, over):
    kwargs = dict(arm=ARM_A, nb=1024, ep=150, c=0.3)
    kwargs.update(over)
    with pytest.raises(ma.ArtifactError):
        ma.write_model_artifact(out, model_with(1024), repo_root=REPO_ROOT, arm=kwargs["arm"], num_buckets=kwargs["nb"], epochs=kwargs["ep"],
                                relative_step_c=kwargs["c"], learning_rate=rate(0.3), manifest_sha256=MANIFEST, training=training())
    assert nothing_written(out)


@pytest.mark.parametrize("over", [dict(learning_rate=0.5), dict(learning_rate=float("nan")), dict(learning_rate=0.0), dict(learning_rate=True),
                                  dict(learning_rate=rate(0.3) * (1 + 1e-12)), dict(manifest_sha256="short"), dict(manifest_sha256=None),
                                  dict(training=training(L=30.0)), dict(training=training(lambda_max=0.0, L=0.0)), dict(training=training(lambda_method="x")),
                                  dict(training=training(train_ids_sha256="z")), dict(training=training(vocabulary_sha256="z")),
                                  dict(training=training(max_length=0)), dict(training={"train_ids_sha256": "a" * 64}),
                                  dict(training={**training(), "labels": [0, 1]}), dict(training={**training(), "extra": 1}), dict(training=None)])
def test_invalid_learning_rate_manifest_or_training_provenance_is_rejected(out, over):
    with pytest.raises(ma.ArtifactError):
        write(out, model_with(1024), **over)
    assert nothing_written(out)


# -- loading and verification -----------------------------------------------------------------------------------------------------------------------


def test_load_reports_metadata_and_verifies_everything(out):
    written = write(out, model_with(1024))
    loaded = load(out, written, expected_model_sha256=ma.model_source_sha256())
    assert loaded.candidate_id == rn.canonical_candidate_id(ARM_A, 1024, 150, 0.3) and loaded.arm == ARM_A and loaded.sha256 == written.sha256
    assert loaded.metadata["manifest_sha256"] == MANIFEST and loaded.metadata["candidate"]["learning_rate"] == rate(0.3)
    expected_fp = ma.sr.model_artifact_fingerprint(model_with(1024))
    assert load(out, written, expected_fingerprint=expected_fp).fingerprint == expected_fp


@pytest.mark.parametrize("bad", [None, "", "abc", "A" * 64, 5])
def test_an_expected_digest_from_outside_is_required(out, bad):
    written = write(out, model_with(1024))
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written, expected_sha256=bad)


def test_wrong_digest_missing_file_and_flipped_bytes(out, tmp_path):
    written = write(out, model_with(1024))
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written, expected_sha256="0" * 64)
    other_dir = tmp_path / "empty"
    other_dir.mkdir()
    with pytest.raises(ma.ArtifactVerificationError):
        load(other_dir, written)
    path = Path(written.path)
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written)


def test_wrong_expected_identity_is_rejected(out):
    written = write(out, model_with(1024))
    cid_b = rn.canonical_candidate_id(ARM_B, 1024, 150, 0.3)
    with pytest.raises(ma.ArtifactVerificationError):  # a different arm: the derived path does not exist / belongs to another candidate
        load(out, written, arm=ARM_B)
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written, expected_candidate_id=rn.canonical_candidate_id(ARM_A, 1024, 150, 0.8))
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written, expected_manifest_sha256="d" * 64)
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written, expected_learning_rate=rate(0.3) * 1.0000001)
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written, expected_learning_rate=None)  # type: ignore[arg-type]
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written, expected_arm=ARM_B)  # candidate does not belong to the expected arm
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written, expected_arm="D_other")
    assert cid_b != written.candidate_id


def test_an_artifact_for_another_candidate_cannot_be_substituted(out):
    first = write(out, model_with(1024, seed=1), ep=50)
    second = write(out, model_with(1024, seed=2), ep=150)
    os.chmod(second.path, stat.S_IWRITE | stat.S_IREAD)
    Path(second.path).write_bytes(Path(first.path).read_bytes())  # the 50-epoch artifact placed at the 150-epoch candidate's path
    digest = mf.sha256_hex(Path(first.path).read_bytes())
    with pytest.raises(ma.ArtifactVerificationError) as caught:
        load(out, second, ep=150, expected_sha256=digest)
    assert "candidate_id" in str(caught.value)


def test_a_changed_model_source_hash_is_detected(out, monkeypatch):
    written = write(out, model_with(1024))
    digest = rewrite(written, lambda a: a.update(model_source_sha256="9" * 64))
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written, expected_sha256=digest)
    written2 = write(out, model_with(2048), nb=2048)
    monkeypatch.setattr(ma, "model_source_sha256", lambda: "8" * 64)  # the live model.py changed after the artifact was written
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written2, nb=2048)
    monkeypatch.undo()
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written2, nb=2048, expected_model_sha256="7" * 64)  # the manifest's recorded model hash differs


def test_a_wrong_expected_fingerprint_is_detected(out):
    written = write(out, model_with(1024, seed=1))
    other = ma.sr.model_artifact_fingerprint(model_with(1024, seed=2))
    with pytest.raises(ma.ArtifactVerificationError):
        load(out, written, expected_fingerprint=other)


FORGERIES = [  # edits that keep the file self-consistent in digest terms
    ("arm_changed", lambda a: a.update(arm=ARM_B)),
    ("candidate_id_changed", lambda a: a["candidate"].update(candidate_id=rn.canonical_candidate_id(ARM_A, 1024, 150, 0.8))),
    ("manifest_changed", lambda a: a.update(manifest_sha256="d" * 64)),
    ("lr_changed", lambda a: a["candidate"].update(learning_rate=0.5, learning_rate_hex=(0.5).hex())),
    ("lr_hex_changed", lambda a: a["candidate"].update(learning_rate_hex=(0.5).hex())),
    ("L_changed", lambda a: a["training"].update(L=40.0)),
    ("lambda_changed", lambda a: a["training"].update(lambda_max=100.0)),
    ("authorization_claimed", lambda a: a["authorization"].update(asserted=True)),
    ("authorization_removed", lambda a: a.pop("authorization")),
    ("extra_field", lambda a: a.update(extra="x")),
    ("missing_model", lambda a: a.pop("model")),
    ("labels_field", lambda a: a["training"].update(labels=[0, 1])),
    ("predictions_field", lambda a: a.update(predictions=[0.5])),
    ("test_field", lambda a: a.update(test_ap=0.5)),
    ("version_changed", lambda a: a.update(artifact_version="model-artifact-2")),
    ("kind_changed", lambda a: a.update(kind="other")),
    ("input_repr_changed", lambda a: a["input_representation"].update(arm=ARM_C)),
    ("limitations_changed", lambda a: a["limitations"].pop()),
    ("bad_base64", lambda a: a["model"].update(weights_b64_chunks=["!!!notbase64!!!"])),
    ("short_weights", lambda a: a["model"].update(weights_b64_chunks=[base64.b64encode(b"\x00" * 16).decode()])),
    ("chunk_layout_changed", lambda a: a["model"].update(weights_b64_chunks=[a["model"]["weights_b64_chunks"][0][:2000], a["model"]["weights_b64_chunks"][0][2000:], *a["model"]["weights_b64_chunks"][1:]])),
    ("chunks_not_list", lambda a: a["model"].update(weights_b64_chunks="abcd")),
    ("chunks_empty", lambda a: a["model"].update(weights_b64_chunks=[])),
    ("chunk_not_string", lambda a: a["model"].update(weights_b64_chunks=[5])),
    ("weights_hash_changed", lambda a: a["model"].update(weights_sha256="0" * 64)),
    ("bias_not_hex", lambda a: a["model"].update(bias_hex="zz")),
    ("bias_nan", lambda a: a["model"].update(bias_hex="nan")),
    ("bias_inf", lambda a: a["model"].update(bias_hex="inf")),
    ("fingerprint_changed", lambda a: a["model"]["fingerprint"].update(bias_hex=(0.0).hex())),
    ("not_trained", lambda a: a["model"].update(is_trained=False)),
    ("header_buckets_changed", lambda a: a["model"].update(num_buckets=2048)),
    ("dtype_changed", lambda a: a["model"].update(weights_dtype="float32")),
]


@pytest.mark.parametrize("name,forge", FORGERIES, ids=[n for n, _ in FORGERIES])
def test_forged_artifacts_with_a_matching_digest_are_still_rejected(out, name, forge):
    written = write(out, model_with(1024))
    digest = rewrite(written, forge)
    with pytest.raises(ma.ArtifactError):
        load(out, written, expected_sha256=digest)


def test_non_finite_weights_inside_a_self_consistent_artifact_are_rejected(out):
    written = write(out, model_with(1024))

    def poison(a):
        w = np.frombuffer(base64.b64decode("".join(a["model"]["weights_b64_chunks"])), dtype="<f8").copy()
        w[10] = np.nan
        raw = w.astype("<f8").tobytes()
        a["model"]["weights_b64_chunks"] = ma._chunks(base64.b64encode(raw).decode())
        a["model"]["weights_sha256"] = mf.sha256_hex(raw)

    digest = rewrite(written, poison)
    with pytest.raises(ma.ArtifactError):
        load(out, written, expected_sha256=digest)


@pytest.mark.parametrize("raw", [b"", b"[]", b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{"a":Infinity}\n', b'{"a":1e999}\n', b'{"a": 1}\n', b'{"a":1}',
                                 b'{"artifact_version":"model-artifact-1",', b"\xff\xfe", pytest.param(b"x" * (ma.MAX_ARTIFACT_BYTES + 1), id="oversized")])
def test_malformed_partial_or_oversized_files_are_rejected_even_with_a_matching_digest(out, raw):
    written = write(out, model_with(1024))
    path = Path(written.path)
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    path.write_bytes(raw)
    with pytest.raises(ma.ArtifactError):
        load(out, written, expected_sha256=mf.sha256_hex(raw))


def test_non_canonical_but_equivalent_json_is_rejected(out):
    written = write(out, model_with(1024))
    path = Path(written.path)
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    pretty = json.dumps(json.loads(path.read_bytes()), indent=2).encode()
    path.write_bytes(pretty)
    with pytest.raises(ma.ArtifactSchemaError):
        load(out, written, expected_sha256=mf.sha256_hex(pretty))


def test_errors_never_echo_parameter_values_or_secrets(out):
    secret = "SECRET-VALUE-424242"
    messages = []
    for call in (lambda: write(out, model_with(1024), manifest_sha256=secret), lambda: write(out, model_with(1024), arm=secret),
                 lambda: write(out, model_with(1024), training={**training(), "train_ids_sha256": secret}),
                 lambda: ma.artifact_filename(MANIFEST, secret), lambda: load(out, ma.WrittenArtifact("p", "0" * 64, 1, "x"), expected_sha256=secret)):
        with pytest.raises(ma.ArtifactError) as caught:
            call()
        messages.append(str(caught.value))
    nan_model = model_with(1024)
    nan_model.weights[0] = np.nan
    nan_model.bias = 0.123456789
    with pytest.raises(ma.ArtifactError) as caught:
        write(out, nan_model)
    messages.append(str(caught.value))
    assert all(secret not in m and "0.123456789" not in m and "nan" not in m.lower().replace("finite", "") for m in messages)


def test_no_labels_scores_or_request_content_in_artifacts(out):
    written = write(out, model_with(1024))
    text = Path(written.path).read_text(encoding="ascii")
    art = json.loads(text)
    problems: list[str] = []
    ma.sr._scan(art, "$", problems)
    assert problems == []
    assert set(art["training"]) == ma._TRAINING_KEYS and set(art["model"]) == ma._MODEL_KEYS and set(art) == ma._TOP_KEYS  # nothing beyond the schema
    assert art["authorization"]["asserted"] is False and "authoriz" in art["authorization"]["note"]


# -- all candidates -----------------------------------------------------------------------------------------------------------------------------------


def candidates():
    for arm in ARMS:
        for nb in (1024, 2048, 4096):
            for ep in (50, 150, 400):
                for c in (0.1, 0.3, 0.8):
                    yield arm, nb, ep, c


def test_expected_names_cover_exactly_81_distinct_files():
    names = ma.expected_artifact_names(MANIFEST)
    assert len(names) == 81 and len(set(names.values())) == 81
    assert set(names) == {rn.canonical_candidate_id(a, nb, ep, c) for a, nb, ep, c in candidates()}
    assert ma.expected_artifact_names("d" * 64) != names  # another manifest, another file set


def test_all_81_independent_candidates_can_be_stored_and_reloaded(out):
    models, digests, rates = {}, {}, {}
    for k, (arm, nb, ep, c) in enumerate(candidates()):
        cid = rn.canonical_candidate_id(arm, nb, ep, c)
        models[cid] = model_with(nb, seed=k)
        written = write(out, models[cid], arm=arm, nb=nb, ep=ep, c=c)
        digests[cid], rates[cid] = written.sha256, rate(c)
    assert len(list(out.iterdir())) == 81
    loaded = ma.load_candidate_set(out, expected_manifest_sha256=MANIFEST, expected_digests=digests, expected_learning_rates=rates)
    assert set(loaded) == set(models)
    sample = inputs()
    for cid, item in loaded.items():
        assert item.model.weights.tobytes() == models[cid].weights.tobytes() and item.model.bias == models[cid].bias
        assert numerics.raw_logits(item.model, sample) == numerics.raw_logits(models[cid], sample)
    assert len({item.fingerprint["weights_sha256"] for item in loaded.values()}) == 81
    first, second = sorted(loaded)[:2]
    assert loaded[first].model is not loaded[second].model


def test_a_candidate_set_with_missing_extra_or_substituted_artifacts_is_rejected(out):
    digests, rates = {}, {}
    for k, (arm, nb, ep, c) in enumerate(candidates()):
        cid = rn.canonical_candidate_id(arm, nb, ep, c)
        digests[cid] = write(out, model_with(nb, seed=k), arm=arm, nb=nb, ep=ep, c=c).sha256
        rates[cid] = rate(c)
    kw = dict(expected_manifest_sha256=MANIFEST, expected_learning_rates=rates)
    ids = sorted(digests)
    with pytest.raises(ma.ArtifactVerificationError):
        ma.load_candidate_set(out, expected_digests={k: v for k, v in digests.items() if k != ids[0]}, **kw)
    with pytest.raises(ma.ArtifactVerificationError):
        ma.load_candidate_set(out, expected_digests={**digests, "extra": "0" * 64}, **kw)
    with pytest.raises(ma.ArtifactVerificationError):
        ma.load_candidate_set(out, expected_digests={**digests, ids[3]: digests[ids[4]]}, **kw)  # digest of a different candidate's file
    victim, donor = ids[5], ids[6]
    victim_path, donor_path = out / ma.artifact_filename(MANIFEST, victim), out / ma.artifact_filename(MANIFEST, donor)
    os.chmod(victim_path, stat.S_IWRITE | stat.S_IREAD)
    victim_path.write_bytes(donor_path.read_bytes())
    with pytest.raises(ma.ArtifactVerificationError):
        ma.load_candidate_set(out, expected_digests=digests, **kw)
    os.chmod(victim_path, stat.S_IWRITE | stat.S_IREAD)
    victim_path.unlink()
    with pytest.raises(ma.ArtifactVerificationError):
        ma.load_candidate_set(out, expected_digests=digests, **kw)


def test_learning_rates_can_be_taken_from_a_verified_manifest_dictionary():
    manifest = {"numerics": {"candidates": [{"candidate_id": "A_request_only|nb=01024|ep=0050|c=0.10", "learning_rate": 0.003}]}}
    assert ma.learning_rates_from_manifest(manifest) == {"A_request_only|nb=01024|ep=0050|c=0.10": 0.003}
    for bad in ({}, {"numerics": {}}, None):
        with pytest.raises(ma.ArtifactSchemaError):
            ma.learning_rates_from_manifest(bad)  # type: ignore[arg-type]


# -- scope and hygiene ----------------------------------------------------------------------------------------------------------------------------------


def test_no_unsafe_deserialization_training_scoring_or_authorization_code():
    text = Path(ma.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
    assert not imported & {"pickle", "cPickle", "marshal", "shelve", "dill", "yaml", "joblib", "subprocess", "random", "ctypes"}
    names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    attrs = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not names & {"eval", "exec", "compile", "__import__"}
    assert not attrs & {"load", "loads", "fit", "predict", "raw_logit", "raw_logits", "average_precision", "request_training_authorization",
                        "request_test_access", "unlink", "remove", "rename", "replace"}
    assert "allow_pickle" not in text and not any(seed in text for seed in ("7101", "7102", "7103"))
    assert not hasattr(ma, "authorize") and not hasattr(ma, "fit_candidate") and not hasattr(ma, "unseal")


def test_runner_gates_are_untouched():
    with pytest.raises(rn.AuthorizationUnavailable):
        rn.request_training_authorization()
    with pytest.raises(rn.AuthorizationUnavailable):
        rn.request_test_access()


def test_no_artifact_files_were_created_in_the_repository():
    assert not list((REPO_ROOT / "tokenization_ai").glob("**/model-artifact-*")) and not list((REPO_ROOT / "tests").glob("**/model-artifact-*"))
    assert ma.source_sha256() == mf.file_sha256(Path(ma.__file__))
