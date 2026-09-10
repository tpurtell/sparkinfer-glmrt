# TrellisMX Spark port baseline

This experimental branch overlays Brandon M. Music's published `runtime/b12x/b12x`
from `brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8` at Hugging Face revision
`db46a5d3b5a8ed0eb0147fcbbd44c690f3918ffe` onto SparkInfer.

The copied runtime sources are governed by the accompanying release licenses,
including the source-available SHAPLEYMCG license; the repository's preexisting
license does not relicense those additions. Source headers remain intact.

This is an integration baseline, not a validated GB10 port or performance claim.
Original SparkInfer files outside the release overlay remain present. No changes
from this branch should be merged into main as part of the recipe work.

The September 9 selected-reference `runtime-reference-20260909/b12x/b12x`
overlay is applied on top of that base. Its source manifest is retained here.
The vLLM reference adapter still requires SM120 and TP4/EP1.
