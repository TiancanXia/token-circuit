# Architecture and extension guide

[English](architecture.md) | [简体中文](architecture.zh-CN.md) · [Documentation](README.md)

## Package boundaries

```text
game.py / start.cmd            stable launch entry
tokenwalk/
  app.py                      Tk lifecycle and input latching
  i18n.py                     paired UI templates and localized stage/result labels
  content/
    guide.py                  read-only bilingual teaching topics and documents
  model/
    architecture.py           configuration, stages, explanations
    mathops.py                pure numerical operators
    forward.py                ordered execution and immutable diagnostics
    cache.py                  committed per-layer KV bank and compatibility check
    prompt.py                 bilingual quote, demonstration tokenization, vocabulary
  simulation/
    maps.py                   doors, roads, grouping, walkable floors
    world.py                  macro / micro / detail / flight / birth / done state
    rooms.py                  registered child specifications, visits and parent stack
    flight.py                 route/timing; emits remaining stage indices
    transfer.py               deterministic dissolve/drift/assemble positions
    navigation.py             pure stage-column/cursor selection
  view/
    painting.py / glyphs.py    Canvas adapter and shared pixel character
    map_view.py                devices and actor drawing
    room_view.py               shared floor/checkpoint drawing at both interior levels
    module_views.py            room-diagram dispatch and non-attention diagrams
    charts.py                  reusable numeric bars/matrices
    attention.py               attention module overview, including rectangular cached Q/K
    attention_glyphs.py        shared head grouping, packed vectors and actual mask grid
    detail_views.py            registered child diagrams from real computation traces
    navigation.py             shared bilingual stage picker drawing
    output.py                 logits/probability diagrams and Top-N summary
    effects.py                 trail, fade and particle drawing
    renderer.py                scene composition and viewport transforms
    hud.py                     compact sidebar and sentence rail
    context.py                 shared selected-stage resolution
    guide_window.py            scrollable read-only Tk model handbook
tools/                        development-only checks/previews
tests/                        numerical, simulation and Tk regression checks
assets/previews/              generated Canvas reconstructions
```

Model code imports no UI or simulation. Simulation consumes model results and emits timing/positions; it imports no Tk and uses caller-provided dt. View consumes state and never performs model steps. app.py owns window focus, held keys, timers and restart.

Teaching content is a separate read-only layer: `content/guide.py` declares paired topic data and constructs configuration snapshots without running operators. `view/guide_window.py` owns the directory, tagged text and scroll widgets; it does not import Session or execute a stage. `view/context.py` resolves the same selected stage for the sidebar and handbook. `view/hud.py` keeps sidebar and sentence-rail layout out of scene composition. Uncomputed selected stages show waiting data, never the previous stage's vector.

The handbook opens from the sidebar's `model-guide` tag. The main map opens the architecture overview; interiors open the selected operation. Game owns one handbook at a time and pauses simulation for its lifetime, independently of focus events. Root actions cannot restart, change configuration, jump or process nodes behind it. Closing clears held keys and resets timing before restoring canvas focus; Esc belongs to the document while it is open. Reading does not advance quote cells, cache publication or child checkpoint progress. Footer space is reserved before the expanding Text widget, so minimum-size windows retain the close/read instructions.

Scene copy uses short paired `brief.*` templates and larger labels. Full explanations, calculation notation, dimensions and limitations live in Topic data. Norm/residual stage variants share guide topics through aliases. Add a topic (or an alias) for any new stage and keep bilingual content together; no window/input branch is needed. Numerical diagrams keep necessary position/crop labels. Large MoE banks prioritize selected experts in the four visible slots and label the omitted expert count and routing-weight mass.

## Execution and state

Manual E and automatic F share `Session._execute_step`; `ForwardPass.execute` rejects skipped/repeated stages. Flight builds an itinerary from the current next_index, follows the same road graph used by the floor/renderer, and emits each remaining index once. Repeated block rooms are collapsed visually, never computationally. F inside a room restores the macro position before continuing.

At Sample, guided mode selects the next prescribed character; free mode uses the actual drawn sample. Particle transfer runs Output → Embedding for 1.4 seconds. The context and sentence rail advance once after the transfer, then the session returns to macro. The final prescribed character goes to done without a new token or transfer.

Held action keys survive automatic scene changes until KeyRelease; movement keys are cleared. Window focus loss freezes model/animation progression and clears keys. R always creates a fresh session using the current configuration. Animated effects derive from simulation time so resizing/drawing cannot advance them.

G opens `StageNavigator`, grouped from actual Stage.block values (global stages, then each configured block). It pauses simulation and intercepts movement/E/Esc; R and language changes close it. The view shows up to three columns, following the selected block for longer configurations. Canvas rows carry real stage indices; mouse/keyboard both call the same Game.jump_to.

`Session.jump_to(index)` validates the index before mutation, builds a new ForwardPass with the current context and round seed, and computes only indices below the target. It then cancels pending flight/transfer, enters the target room/checkpoint and stores that room's macro door. Generation/revealed cells remain unchanged. Rebuilding avoids stale residual branches or diagnostics when moving backwards. Ordinary E/F execution resumes through the existing executor.

Output drawing is separated in view/output.py. Unembedding has logits; Output has full-vocabulary Softmax probabilities. `output_summary` only selects display indices and sums the omitted values; it does not renormalize or change the model distribution. The room shows Top 7/actual vocabulary size and omitted probability mass; output sidebar counts vocabulary tokens rather than feature channels. The default vocabulary is 16 in Chinese and 12 in English; custom model sizes are read dynamically.

## Languages and token labels

`Locale.text(key, **values)` reads paired Chinese/English templates, fails on missing keys and validates locale names. Session translates its own status messages; Painter supplies the same locale to shared drawings. Model execution stays locale-independent; original stage titles/formulas remain shared. Existing Chinese stage subtitles live in architecture content; the catalog pairs them with English subtitles without repeating the Chinese text. Numerical diagnostic labels are translated only in the view.

`prompt.demo_config(blocks, language)` constructs the preset vocabulary/context: Chinese characters or English words/punctuation. These are demonstration tokens, not a production tokenizer. `join_tokens` restores spacing for prompt metadata. No future tokens are rendered in the sentence rail. `Painter.fit_text` shrinks single-line text and abbreviates narrow matrix labels; English rail cells are wider.

L / the language button resets the whole session to avoid carrying incompatible vocabularies or a pending output token into another language. Layer configuration is preserved; explicit custom ModelConfig objects are preserved as-is, so language changes only their interface. R preserves the selected locale. Add UI copy to the paired catalog; never copy a scene or numerical handler for another language.

## Extending

- Add network operations in model configuration/execution, with pure numerical tests.
- Map operations to room groups in simulation/maps.py; define any added device door and road edges there. Floor checks, drawing and flight must use this shared graph.
- Add parent room diagrams in view/module_views.py (specialized diagrams may have their own view file) and device art in map_view.py. Use stored room/layer results rather than later global results.
- To add an interior to an existing operation, register a RoomSpec in simulation/rooms.py, pair its explanations in i18n.py, and register its drawer in view/detail_views.py. No new input branch, walking rule or floor renderer is needed. Keep any new numerical diagnostics in the operation’s Result.details.
- Add the stage's short screen copy to i18n.py and detailed teaching topic to content/guide.py; use a topic alias when the explanation is shared. The handbook tests require every stage to have a complete bilingual document.
- Add timed effects as pure geometry/state under simulation and drawing under view; input stays in app.py.
- Keep generated images and optional libraries outside runtime code. Preserve the root launch and verification commands.

Current room layout targets the configured tiny decoder, not arbitrary neural networks. Broader topology support should replace map definitions deliberately rather than silently reusing unsupported rooms.

Runtime Python files use standard Ruff formatting. Keep current reference documents at docs/; superseded design/implementation milestones belong in docs/archive/. Preview artifacts remain under assets/previews/ and optional tools never enter runtime imports.

## Nested interiors and caches

RoomSpec defines checkpoint labels/locations from data. Rooms stores per-stage visits and parent scene/position. E on a reached parent checkpoint opens the child using ForwardPass.preview: stored data for completed stages, or detached current-stage execution for a preview. Preview never changes the real executor, RNG or cache. Local checkpoints advance Visit.progress; only the final checkpoint calls the same Session._execute_step used by ordinary E/F. Re-entering preserves progress; completed visits remain inspectable without replay. M/Escape pops one level (Escape first leaves fullscreen when active); F unwinds both interiors before using the shared road flight. G clears stale visits and rebuilds prerequisites. Tab returns to the module before changing its reached layer; N only selects a diagnostic query head.

ModelConfig separates query heads and KV heads; head_dim=dim/heads and kv_dim=num_kv_heads*head_dim. attention_mode/kv_head_for expose the layout to shared diagrams. The pure attention_trace operator computes grouped-head scores, visibility, probabilities and weighted V once; ordinary attention wraps the same implementation. A query at the final valid key needs no explicit mask; earlier queries mask future scores with -infinity. Per-row mask_applied diagnostics drive the view, including chunked-query cases. Projection matrices are stored output×input, hence row-vector formulas use X Wᵀ.

MHA/GQA configure the whole Attention layer. QKV’s head checkpoint displays the mapping; the Head attention operation follows RoPE/KV preparation and ends at the unique Concat child checkpoint. Output · W_o inspects packed input and projects it. Phase metadata is captured in each Result (prefill/decode/full), including detached previews whose cache publisher is disabled. Room labels specialize from actual mask diagnostics rather than a global cache switch. Parent views show only the selected stage’s result, or a labeled static architecture layout while it waits; they never substitute another stage’s diagram.

KVCache holds only fully committed per-layer rotated K/unrotated V, a context tuple and structural model signature. ForwardPass snapshots reusable history only when the new context appends exactly one token to the committed prefix with an unchanged model. Incremental positions retain their absolute prefix index. Each block stages its new combined K/V separately; Sample publishes all layers together. Changed/non-append/truncated contexts recompute rather than pretending that cached hidden states remain valid. The default Game enables cache; standalone Session defaults to full-prefix execution for existing callers, and accepts cache_enabled=True. Jumping rebuilds current-round prerequisites while retaining compatible committed history, so a jump into Attention can still inspect real cache reuse. If history is incompatible, it recomputes the whole prefix. It publishes a rebuilt bank only when that forward subsequently finishes.

All child drawings consume snapshots, never compute model steps. The common room_checkpoints/checkpoint_complete/checkpoint_title interface drives both interior floors. Current support is deliberately three map levels with arbitrary registered module interiors, rather than a general recursive game engine.

## Verification

The test suite covers full-prefix versus incremental-cache probabilities, GQA versus explicitly expanded K/V, causal masks, detached previews, cache compatibility, child progress/returns/revisits, flight/jump cancellation, held settings keys, bilingual child diagrams and cropped weighted-sum contributions. Run the root verification commands after changes. tools.attention_preview additionally reconstructs both languages from an actual six-token cached decode, with a cache-off comparison; it requires optional Pillow and does not affect game dependencies.

Canvas reconstructions and hidden Tk event checks complement native-window checks. They do not establish that physical desktop keyboard input was manually tested.

```sh
python -m unittest discover -s tests -v
python check_ui.py
python -m ruff check tokenwalk game.py check_ui.py preview.py --select F
python -m ruff format --check tokenwalk
```

Current screenshots and animation used by the READMEs live in assets/showcase/ and are tracked. Bulk generated previews and historical planning documents remain local and ignored. See [CONTRIBUTING.md](../CONTRIBUTING.md) for synchronization and publishing workflow.
