"""Config sweep + 100-phrase Spanish VAD benchmark.

Finds the combination of question wording, state shape and thresholds
that best detects turn-end, interruptions and backchannels.

Structure:
  1. DATASET: 100 labeled Spanish partials across 5 scenarios
     (turn_complete, mid_turn, barge_correction, backchannel, side_talk).
  2. QUESTION SETS: variants of floor/barge question wording (Spanish).
  3. STATE SHAPES: how the partial+context is rendered into the state
     the model sees (raw text, dict fields, instruction prefixes).
  4. run(): score every (questions, state_shape, threshold) combo ->
     accuracy + F1 per class + confusion; print a ranked table and the
     winning config as copy-paste constants.

Usage:
  python debug_sweep.py              # full sweep (~several minutes)
  python debug_sweep.py --quick      # 25-phrase sample, fewer configs
  python debug_sweep.py --best       # run only the winning combo
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from laya.onnx_agent import ONNXAgent

from jev_vad._download import ensure_model_dir
from jev_vad.detector import load_agent

# ---------------------------------------------------------------------------
# 1. Dataset: 100 labeled Spanish partials.
# Label is the ground truth for what the voice pipeline should do.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One benchmark phrase with its ground-truth pipeline action."""

    text: str
    label: str  # turn_complete | mid_turn | take_floor | backchannel | side_talk
    agent_context: str = ""  # what the agent was (or had been) saying


def _turn_samples() -> list[Sample]:
    """Agent-silent scenarios: complete turns vs mid-turn fragments."""
    complete = [
        "quiero reservar un vuelo a París para mañana por la mañana",
        "dame dos billetes para el tren de las ocho",
        "¿qué tiempo va a hacer el fin de semana en Málaga?",
        "resuérvame una ecuación de segundo grado",
        "pon música de jazz suave para estudiar",
        "recuérdame llamar al dentista el martes a las cinco",
        "quiero pedir una pizza margarita grande",
        "¿a qué hora abre la farmacia de la plaza mayor?",
        "necesito un taxi para el aeropuerto a las seis",
        "apaga la calefacción de la sala de estar",
        "quiero devolver este paquete, llegó roto",
        "traduce al inglés 'la casa azul'",
        "pon un temporizador de diez minutos",
        "¿cuánto cuesta enviar una carta certificada?",
        "busca el restaurante más cercano que tenga mesas libres",
        "vamos a ver la nueva película de Almodóvar",
        "quiero cancelar mi suscripción mensual",
        "déjame un hueco en la agenda para el viernes",
        "¿me puedes decir el resultado del partido de ayer?",
        "sí, perfecto, gracias",
        "no, eso es todo, muchas gracias",
        "vale, va a ser que no",
        "mm, no tengo más preguntas",
        "no, nada, gracias por la ayuda",
        "vale, perfecto, hasta luego",
    ]
    mid = [
        "quiero reservar un vuelo a París para",
        "dame dos billetes para el tren de",
        "¿qué tiempo va a hacer el fin de",
        "resuérvame una ecuación",
        "pon música de",
        "recuérdame llamar al",
        "quiero pedir una pizza",
        "¿a qué hora abre la",
        "necesito un taxi para el",
        "apaga la",
        "quiero devolver este",
        "traduce al inglés",
        "pon un temporizador",
        "¿cuánto cuesta enviar",
        "busca el restaurante más cercano que",
        "vamos a ver la nueva",
        "quiero cancelar mi",
        "déjame un hueco en la",
        "¿me puedes decir el",
        "eh... a ver, a ver",
        "bueno, es que yo lo que quiero es",
        "mm, a ver, déjame pensar",
        "o sea, yo lo que estaba pensando era",
        "eh, no sé, quizá que",
        "a ver cómo te lo digo yo",
    ]
    ctx = "¿En qué puedo ayudarte?"
    return [Sample(t, "turn_complete", ctx) for t in complete] + [
        Sample(t, "mid_turn", ctx) for t in mid
    ]


def _barge_samples() -> list[Sample]:
    """Agent-speaking scenarios: corrections, backchannels, side-talk."""
    ctx_conf = (
        "Su reserva está confirmada, referencia ABX123. Recibirá un "
        "correo electrónico con los detalles en unos minutos."
    )
    ctx_choice = "Tengo vuelos el jueves y el viernes por la mañana, ¿cuál prefiere?"
    ctx_info = (
        "El vuelo tiene una duración de dos horas y diez minutos, y "
        "incluye una maleta de mano en la tarifa básica."
    )
    corrections = [
        (ctx_conf, "espera, no, me he equivocado de fecha"),
        (ctx_conf, "no, no, ese no es mi número de referencia"),
        (ctx_choice, "perdón, quería un tren en realidad"),
        (ctx_choice, "no, al revés, el del jueves"),
        (ctx_info, "un momento, ¿eso incluye la maleta facturada?"),
        (ctx_choice, "calla calla, que se me ha olvidado decirte"),
        (ctx_info, "perdona que te interrumpa, ¿a qué terminal sale?"),
        (ctx_conf, "no, esa no es mi tarjeta"),
        (ctx_choice, "¡eh!, no, el viernes no puedo"),
        (ctx_conf, "espérate, ¿y si quiero cambiar el nombre?"),
        (ctx_info, "¿cuánto cuesta la maleta facturada?"),
        (ctx_choice, "el del jueves, el del jueves"),
        (ctx_conf, "no, no me llega ningún correo"),
        (ctx_choice, "mire, mejor el más barato"),
        (ctx_info, "oiga, y ¿hay wifi en el vuelo?"),
        (ctx_conf, "que no, que yo no he reservado nada"),
        (ctx_choice, "va a ser que ninguno de los dos"),
        (ctx_info, "¿y los niños pagan lo mismo?"),
        (ctx_conf, "para, para, que la fecha está mal"),
        (ctx_choice, "no, espera, déjame mirar la agenda"),
    ]
    backchannels = [
        (ctx_conf, "sí"),
        (ctx_conf, "vale"),
        (ctx_conf, "de acuerdo"),
        (ctx_conf, "perfecto"),
        (ctx_conf, "muy bien"),
        (ctx_choice, "sí sí"),
        (ctx_choice, "vale vale"),
        (ctx_info, "ya"),
        (ctx_info, "ya, ya"),
        (ctx_info, "ajá"),
        (ctx_info, "mhm"),
        (ctx_info, "de acuerdo, de acuerdo"),
        (ctx_conf, "genial"),
        (ctx_conf, "estupendo"),
        (ctx_info, "bueno, vale"),
        (ctx_info, "eh, sí"),
        (ctx_conf, "vale, gracias"),
        (ctx_info, "sí, entiendo"),
        (ctx_info, "uhm"),
        (ctx_info, "claro, claro"),
    ]
    side = [
        (ctx_conf, "oye, mamá, ¡ya voy!"),
        (ctx_conf, "juan, apaga la tele"),
        (ctx_info, "cariño, ¿has visto las llaves?"),
        (ctx_info, "un segundo, cariño, que estoy con el ordenador"),
        (ctx_choice, "¿te pongo un café, Marta?"),
        (ctx_conf, "perdona, hijo, ahora voy"),
        (ctx_info, "eh, ¿quién llama a la puerta?"),
        (ctx_conf, "un momento, que suena el timbre"),
        (ctx_info, "luego te digo, ahora no puedo"),
        (ctx_choice, "papá, ven un momento"),
    ]
    out = []
    for ctx, t in corrections:
        out.append(Sample(t, "take_floor", ctx))
    for ctx, t in backchannels:
        out.append(Sample(t, "backchannel", ctx))
    for ctx, t in side:
        out.append(Sample(t, "side_talk", ctx))
    return out


def dataset() -> list[Sample]:
    """The 100-phrase benchmark."""
    return _turn_samples() + _barge_samples()


# ---------------------------------------------------------------------------
# 2. Question-set variants. Same ids as the PoC (barge/floor) so the
#    results map 1:1 onto questions.py.
# ---------------------------------------------------------------------------


def _q_floor_v1() -> dict:
    """Current PoC wording (baseline)."""
    return {
        "floor": {
            "type": "choice",
            "instructions": (
                "¿Es `partial_transcript` un enunciado completo, o una "
                "transcripción parcial todavía en curso?"
            ),
            "criteria": {
                "turn_complete": (
                    "El enunciado forma una petición o afirmación "
                    "completa y autocontenida; el hablante ha terminado "
                    "y espera respuesta"
                ),
                "still_speaking": (
                    "El enunciado es un fragmento incompleto; el "
                    "hablante está en mitad de una frase y se esperan "
                    "más palabras"
                ),
            },
        }
    }


def _q_floor_v2() -> dict:
    """ASR-partial framing: emphasise the word stream keeps growing."""
    return {
        "floor": {
            "type": "choice",
            "instructions": (
                "Esto es lo que se lleva transcrito de lo que dice el "
                "usuario (`partial_transcript`). ¿Ha terminado de "
                "hablar, o la transcripción seguirá creciendo?"
            ),
            "criteria": {
                "turn_complete": (
                    "ha terminado de hablar: lo transcrito ya se "
                    "entiende como una petición o respuesta completa"
                ),
                "still_speaking": (
                    "sigue hablando: la frase está a medias y "
                    "faltan palabras por llegar"
                ),
            },
        }
    }


def _q_floor_v3() -> dict:
    """Action framing: decide as the pipeline would."""
    return {
        "floor": {
            "type": "choice",
            "instructions": (
                "Eres el detector de fin de turno de un asistente de "
                "voz. Con solo lo transcrito hasta ahora, ¿deberías "
                "responder ya al usuario o esperar a que siga "
                "hablando?"
            ),
            "criteria": {
                "turn_complete": (
                    "responder ya: lo transcrito basta para entender la petición"
                ),
                "still_speaking": (
                    "esperar: la frase está incompleta y faltan palabras"
                ),
            },
        }
    }


def _q_barge_v1() -> dict:
    """English framing, Spanish state — the sweep winner by a wide
    margin over the all-Spanish variant (mmBERT task framing is
    English-native; measured ΔF1 ≈ +0.3 on take_floor)."""
    return {
        "barge": {
            "type": "choice",
            "instructions": (
                "The assistant just said `agent_context`. The user "
                "interrupted with `partial_transcript`. Is the user "
                "taking the floor from the assistant?"
            ),
            "criteria": {
                "take_floor": (
                    "a deliberate interjection or correction directed "
                    "at the assistant; the agent should stop and listen"
                ),
                "backchannel": (
                    "a brief acknowledgment or filler; the user is not "
                    "trying to stop the agent"
                ),
                "side_talk": (
                    "the user is talking to someone else, not to the assistant"
                ),
            },
        }
    }


def _q_barge_v2() -> dict:
    """All-Spanish variant (comparison)."""
    return {
        "barge": {
            "type": "choice",
            "instructions": (
                "El asistente acaba de decir `agent_context`. El "
                "usuario le ha interrumpido con `partial_transcript`. "
                "¿Está el usuario tomando la palabra?"
            ),
            "criteria": {
                "take_floor": (
                    "una interrupción deliberada o corrección dirigida "
                    "al asistente; el asistente debería callarse y "
                    "escuchar"
                ),
                "backchannel": (
                    "un breve reconocimiento o muletilla; el usuario "
                    "no intenta detener al asistente"
                ),
                "side_talk": (
                    "el usuario está hablando con otra persona, no con el asistente"
                ),
            },
        }
    }


QUESTION_SETS: dict[str, Callable[[], dict]] = {
    "floor_v1": _q_floor_v1,
    "floor_v2": _q_floor_v2,
    "floor_v3": _q_floor_v3,
    "barge_v1": _q_barge_v1,
    "barge_v2": _q_barge_v2,
}


# ---------------------------------------------------------------------------
# 3. State shapes. fn(sample, detector) -> state the model sees.
# ---------------------------------------------------------------------------

STATE_SHAPES: dict[str, Callable[[Sample, Any], Any]] = {
    # bare dict with the two fields (PoC default)
    "dict": lambda s, d: (
        {
            "agent_context": d._tail(s.agent_context),
            "partial_transcript": d._tail(s.text),
        }
        if s.agent_context
        else {"partial_transcript": d._tail(s.text)}
    ),
    # single raw string, just the partial
    "raw": lambda s, d: d._tail(s.text),
    # dict + an explicit speaker turn list (chat-style)
    "turns": lambda s, d: (
        [
            *(
                [{"role": "assistant", "content": d._tail(s.agent_context)}]
                if s.agent_context
                else []
            ),
            {"role": "user", "content": d._tail(s.text)},
        ]
    ),
}


# ---------------------------------------------------------------------------
# 4. Sweep machinery.
# ---------------------------------------------------------------------------


@dataclass
class ConfigResult:
    """Metrics for one (question_set, state_shape) config."""

    question_set: str
    state_shape: str
    per_label_correct: Counter = field(default_factory=Counter)
    per_label_total: Counter = field(default_factory=Counter)
    confusion: Counter = field(default_factory=Counter)
    latencies: list[float] = field(default_factory=list)
    # raw probabilities keyed by sample index, for threshold tuning
    prob_trace: list[dict[str, float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Fraction of correct top-1 decisions."""
        total = sum(self.per_label_total.values())
        correct = sum(self.per_label_correct.values())
        return correct / total if total else 0.0

    def f1(
        self, label: str, prob_key: str | None = None, threshold: float | None = None
    ) -> float:
        """F1 for one class, argmax or thresholded on `prob_key`."""
        if threshold is not None and prob_key:
            tp = fp = fn = 0
            for lbl, probs in zip(self.labels, self.prob_trace, strict=True):
                pred = probs.get(prob_key, 0.0) >= threshold
                if label in ("mid_turn",):
                    truth = lbl == label
                else:
                    truth = lbl == label
                tp += pred and truth
                fp += pred and not truth
                fn += (not pred) and truth
        else:
            tp = fp = fn = 0
            for lbl, probs in zip(self.labels, self.prob_trace, strict=True):
                # prob_trace keys are '<qid>.<option>' — for this class
                # look for the option that carries its semantics: the
                # floor qid for turn labels, barge.take_floor otherwise.
                key = next(
                    (k for k in probs if label in k),
                    None,
                )
                pred = max(probs, key=probs.get)  # type: ignore[arg-type]
                if key is not None:
                    pred = (
                        key.split(".", 1)[1]
                        if key.startswith(("floor.", "barge."))
                        else pred
                    )
                tp += pred == label and lbl == label
                fp += pred == label and lbl != label
                fn += pred != label and lbl == label
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _decide(
    agent: ONNXAgent,
    state: Any,
    questions: dict,
    sample_label: str,
    res: ConfigResult,
) -> str:
    """One model pass; record metrics; return the argmax choice."""
    t0 = time.perf_counter()
    out = agent.predict(state, questions)
    res.latencies.append((time.perf_counter() - t0) * 1000)
    answers = out["answers"]
    # record every probability we may later threshold on
    probs: dict[str, float] = {}
    for qid, ans in answers.items():
        for opt, p in ans["probabilities"].items():
            probs[f"{qid}.{opt}"] = p
    res.prob_trace.append(probs)
    res.labels.append(sample_label)
    for qid, ans in answers.items():
        pred = ans["choice"]
        truth = sample_label
        res.confusion[(truth, pred)] += 1
        res.per_label_total[truth] += 1
        res.per_label_correct[truth] += pred == truth
    return out["answers"]["floor" if "floor" in answers else "barge"]["choice"]


def run_config(
    agent: ONNXAgent,
    samples: list[Sample],
    question_set: str,
    state_shape: str,
) -> ConfigResult:
    """Score one config over the dataset."""
    questions = QUESTION_SETS[question_set]()
    shape_fn = STATE_SHAPES[state_shape]
    res = ConfigResult(question_set=question_set, state_shape=state_shape)
    for s in samples:
        state = shape_fn(s, _TailHelper())
        if "floor" in questions and "barge" in questions:
            # combined set: pick the answer matching this scenario
            want = "floor" if s.label in ("turn_complete", "mid_turn") else "barge"
            sub = {want: questions[want]}
        else:
            sub = questions
        _decide(agent, state, sub, s.label, res)
    return res


class _TailHelper:
    """Minimal word-budget helper so state shapes don't need a detector."""

    @staticmethod
    def _tail(text: str, max_words: int = 40) -> str:
        """Keep the last `max_words` words."""
        words = text.split()
        return " ".join(words[-max_words:])


FLOOR_SETS = ["floor_v1", "floor_v2", "floor_v3"]
BARGE_SETS = ["barge_v1", "barge_v2"]
SHAPES = ["dict", "raw", "turns"]


def sweep(samples: list[Sample], verbose: bool = True) -> list[ConfigResult]:
    """Run every relevant (question_set, state_shape) combination."""
    agent = load_agent(ensure_model_dir(None))
    results: list[ConfigResult] = []
    floor_samples = [s for s in samples if s.label in ("turn_complete", "mid_turn")]
    barge_samples = [
        s for s in samples if s.label in ("take_floor", "backchannel", "side_talk")
    ]
    combos = [(qs, sh, floor_samples) for qs in FLOOR_SETS for sh in SHAPES]
    combos += [(qs, sh, barge_samples) for qs in BARGE_SETS for sh in SHAPES]
    for qs, sh, sub in combos:
        res = run_config(agent, sub, qs, sh)
        results.append(res)
        if verbose:
            print(
                f"{qs:<9} {sh:<6} acc={res.accuracy:.2f} "
                f"p50={sorted(res.latencies)[len(res.latencies) // 2]:.0f}ms"
            )
    return results


def tune_thresholds(results: list[ConfigResult]) -> list[tuple]:
    """Grid-search probability thresholds per barge config.

    Returns (config, best_threshold, best_f1_take_floor,
    backchannel_recall_at_threshold) tuples, ranked.
    """
    out = []
    for res in results:
        if not res.question_set.startswith("barge"):
            continue
        best = (0.5, 0.0, 0.0)
        for th in [i / 20 for i in range(3, 17)]:  # 0.15 .. 0.80
            f1_tf = res.f1("take_floor", "barge.take_floor", th)
            if f1_tf > best[1]:
                # recall of backchannel at the same operating point:
                # a decision is "let agent keep talking" when
                # barge.take_floor < th; measure that precision on
                # backchannel+side_talk samples.
                tp = fp = fn = 0
                for lbl, probs in zip(res.labels, res.prob_trace, strict=True):
                    keep = probs.get("barge.take_floor", 0.0) < th
                    truth = lbl != "take_floor"
                    tp += keep and truth
                    fp += keep and not truth
                    fn += (not keep) and truth
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                f1_keep = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                best = (th, f1_tf, f1_keep)
        out.append((res, *best))
    out.sort(key=lambda t: -t[2])
    return out


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small sample")
    parser.add_argument("--best", action="store_true", help="winner only")
    args = parser.parse_args()

    samples = dataset()
    if args.quick:
        samples = samples[:10] + samples[50:65]
    if args.best:
        agent = load_agent(ensure_model_dir(None))
        # winner from the full run: barge_v1/dict, th 0.30 + filler rule
        res = run_config(agent, samples, "barge_v1", "dict")
        print(f"barge_v1+dict acc={res.accuracy:.2f}")
        return

    print(
        f"sweep: {len(samples)} phrases, "
        f"{(len(FLOOR_SETS) + len(BARGE_SETS)) * len(SHAPES)} configs\n"
    )
    t0 = time.perf_counter()
    results = sweep(samples)
    dt = time.perf_counter() - t0

    print(f"\n== ranked by accuracy ({dt:.0f}s total) ==")
    ranked = sorted(results, key=lambda r: -r.accuracy)
    for r in ranked:
        n = sum(r.per_label_total.values())
        p50 = sorted(r.latencies)[len(r.latencies) // 2]
        per = " ".join(
            f"{lbl}={r.per_label_correct[lbl]}/{r.per_label_total[lbl]}"
            for lbl in sorted(r.per_label_total)
        )
        print(
            f"{r.question_set:<9} {r.state_shape:<6} "
            f"acc={r.accuracy:.3f} f1_tc={r.f1('turn_complete'):.2f} "
            f"f1_tf={r.f1('take_floor'):.2f} p50={p50:.0f}ms  {per}"
        )

    print("\n== best barge configs with tuned take_floor threshold ==")
    for res, th, f1_tf, f1_keep in tune_thresholds(results)[:4]:
        print(
            f"{res.question_set}/{res.state_shape}: th={th:.2f} "
            f"F1(take_floor)={f1_tf:.2f} F1(keep-talking)={f1_keep:.2f}"
        )

    best = ranked[0]
    print(f"\nWINNER: {best.question_set}/{best.state_shape} acc={best.accuracy:.3f}")
    print("confusion (truth->pred):")
    for (truth, pred), n in best.confusion.most_common():
        if truth != pred and n:
            print(f"  {truth:<14} -> {pred:<14} {n}")


if __name__ == "__main__":
    main()
