# Annotation codebook

Operationalises embodiment for sentence-level classification of narrative fiction. Used
for manual annotation and, in the form of `prompt.md`, for LLM prompting.

## Annotation unit

- The unit is the sentence, treated as the minimal unit of phenomenological integration
  and narrative coherence.
- Each sentence receives exactly one label.
- Where several embodied modes occur in one sentence, annotate the dominant mode. The
  precedence rules below resolve genuinely ambiguous cases. This is an acknowledged
  simplification, accepted for scalability.

## Label set

| Label | Group | Short definition |
|---|---|---|
| `haptic_human_animal` | Haptics | Physical contact with a person or animal |
| `haptic_material` | Haptics | Physical contact with objects, surfaces, environment (incl. passive contact) |
| `character_move` | Movement | Character changes position or location, no emphasis on contact |
| `sight_human` | Sight | Foregrounded visual perception of people or animals |
| `sight_material` | Sight | Foregrounded visual perception of objects or environment |
| `character_feltness` | Feltness | Character is affected by sensory or affective qualities of the environment |
| `facial_expr` | Expression | Explicit description of a character's facial expression |
| `gesture` | Expression | Communicative or expressive bodily movements |
| `no_rel` | Negative | No embodiment in the narrated situation |
| `book_paratext` | Negative | Non-narrative elements (titles, tables of contents, notes) |
| `segmentation_error` | Negative | Sentence is wrongly segmented (e.g. cut off at "Mrs.", a colon; semicolons excepted) |

## Precedence rules

Apply in order; stop at the first match.

1. **Segmentation error.** An incomplete sentence, cut off at an abbreviation, a colon,
   or otherwise lacking a clear end, is `segmentation_error` before its apparent content
   is considered at all. Semicolons are not segmentation errors.
2. **Paratext.** Non-narrative elements (headings, tables of contents, notes) are
   `book_paratext` regardless of any embodied content they might otherwise contain.
3. **Temporality and modality.** Embodiment counts only if it occurs in the narrated
   situation itself. Recalled, anticipated, planned, hypothetical or conditional events
   are `no_rel`, even where the described action would otherwise be haptic, movement,
   sight or feltness.
4. **Haptics over movement and sight.** Direct bodily contact, active or passive, takes
   precedence. A sentence combining touch with movement or looking is annotated as touch.
5. **Movement over sight.** A change of position without emphasis on contact is
   `character_move`; static position alone is not movement.
6. **Sight only when foregrounded.** Seeing is annotated only when the act of looking is
   itself the point of the sentence. Not when it is folded into movement or touch, and
   not when it merely leads into an action that is annotated instead.
7. **Feltness requires a present, affected character.** Sensory or affective qualities of
   the environment are `character_feltness` only if a character is there to be affected by
   them. Environmental description on its own is `no_rel`.
8. **Expression.** `facial_expr` covers sentences that explicitly describe or refer to a
   character's facial expression. `gesture` covers bodily movements that convey meaning,
   intention or emotion: movements of the hands, arms, head or whole body used
   communicatively or expressively.
9. **Default.** A sentence matching none of the above is `no_rel`.

## Category definitions

### Haptics (touch / physical contact)

The character physically touches, or is in direct bodily contact with, something or
someone. Includes active touch (grasping, pressing, pulling, hitting), passive contact
(leaning, resting against, sitting on), and contact mediated by tools where bodily contact
is implied. Split into `haptic_human_animal` (contact with another person or animal) and
`haptic_material` (contact with objects, surfaces or the environment).

- "She grasped his hands." → `haptic_human_animal`
- "He pressed his palm against the wall." → `haptic_material`
- "Mr. Tartar and Lobley pulled a pair of oars." → `haptic_material`
- "Terrified by this alarming spectacle, Miss Twinkleton placed another shilling in each hand." → `haptic_human_animal`
- "Touching a spring, an iron railing sprang up, about five feet high." → `haptic_material`

Edge case: "He ascends a broken staircase, opens a door, looks into a dark stifling room,
and says: 'Are you alone here?'" Movement, touch and sight are all present; touch (opening
the door) wins under the precedence rule → `haptic_material`.

### Movement

The character changes position or location. Distinguished from haptics by the absence of
(emphasis on) physical interaction. Includes navigation relative to objects, even without
contact.

- "Meanwhile the two gentlemen [...] descended the doorsteps, ascended their carriages, and drove away [...]" → `character_move`
- "Within half an hour they were setting this matter right by going up the river." → `character_move`

### Sight

Visual perception of people or the environment, annotated only when the act of perceiving
is foregrounded from a character's (relative) standpoint. Much seeing happens through
movement and touch; only the relatively static position of a character looking at things,
people or animals is relevant here. Do not annotate sight when seeing occurs as part of
movement or touch, when sight directly leads to an action, or when the sentence merely
describes what is happening and a character only implicitly observes it. Split into
`sight_human` and `sight_material`.

- "Rosamond and Laura looked at her and one another with surprise [...]" → `sight_human`
- "The man glanced up at Stubbs and then turned his attention to Wilson." → `sight_human`
- "She looked at the painting." → `sight_material`
- "She saw the painting and touched it." → `haptic_material` (touch takes precedence)

### Feltness / atmosphere (Stimmung) / affect

Sensory or affective qualities of the environment, including sound, smell, taste,
temperature, texture, density, weight, colour as affective quality, or general atmosphere.
Concerns the sensing or feeling subject: the character is moved or affected by the
surrounding environment. Sight is excluded. Annotate `character_feltness` only when a
character is present and affected by the surroundings.

- "He travels thither by the means by which Rosa travelled, and arrives, as Rosa arrived, on a hot, dusty evening." → `character_feltness`
- "Of course I heard sounds, but even here they were faint [...]" → `character_feltness`
- "But pacing along under the stars, with only the sound of his own footsteps for company, placed a very different light on the matter." → `character_feltness`

### Facial expressions

Sentences that explicitly describe or refer to the facial expressions of characters.

- "If you smile like that again you make me cry." → `facial_expr`
- "'A hundred times more so!' cried Bart, his eyes kindling and a flush suffusing his cheeks." → `facial_expr`
- "Mirabelle Marie gaped speechlessly at him." → `facial_expr`

Edge case: "The lieutenant saluted stiffly, but Chester noted an angry gleam in his eye."
Could be gesture, `sight_human` or facial expression; the dominant mode is annotated.

### Gestures

Bodily movements that convey meaning, intention or emotion: movements of the hands, arms,
head or whole body used communicatively or expressively. Includes deliberate or expressive
movements (nod, shake one's head, wave, point, shrug), actions functioning as non-verbal
communication or emphasising speech, and posture shifts or embodied reactions conveying
attitude or emotion (recoiled, leaned forward eagerly).

- "Poirot lifted his hand." → `gesture`

### Negative categories

`no_rel`: no embodiment in the narrated situation. Includes pure dialogue without embodied
content, abstract statements, and events filtered out by the temporality rule.

- "Do not let me hear of your offending again." → `no_rel`
- "She stood at the door." → `no_rel` (static position; not movement)
- "The walls were rough and uneven." → `no_rel` (environmental description without a perceiving character)

`book_paratext`: non-narrative elements (titles, tables of contents, notes).

`segmentation_error`: sentence produced by faulty sentence splitting (e.g. at "Mrs.", or
at a colon), semicolons excepted.

## Temporality and modality rule

Annotate only embodiment that occurs in the situation actually being described. Events
merely recalled, long past, anticipated, planned, hypothetical or conditional are `no_rel`.

- "When I was a child, my father used to hit me." → `no_rel` (not `haptic_human_animal`)
- "When we get to the next stop, he will come back." → `no_rel` (not `character_move`)
- "If you came, you would understand." → `no_rel`

## Known edge cases

1. **Precedence of gesture and facial_expr relative to the core categories.** "Then the
   lad motioned to Chester to remain silent, and laid his ear to the hole." → `gesture`
2. **Attention without explicit seeing.** "He abandoned the chest and gave his attention to
   the cupboard." → `sight_material`
3. **Sight of non-realistic entities.** "I could see the eyes like blue flames, the face
   with a hideous grin [...]" → `sight_human`
4. **Mixed involuntary movement and contact.** "[H]e sank to the ground under the pressure
   of his brother's hand [...] and crouching in the shadow [...] remained motionless with
   dismay." → `haptic_human_animal`
5. **Feltness in direct speech.** "How dismal and gloomy this night is!" Direct speech, but
   arguably `character_feltness`.
6. **Eyes as body part.** "David shut his eyes again after seeing it." → `facial_expr`
7. **Summarised action.** "During the forenoon we examined the downstairs rooms pretty
   thoroughly." → `no_rel`

## Retired labels

Earlier drafts of the scheme also used `character_cognition` (originally `HUMAN_COGNITION`),
`description_feltness` and `character_static`. These are folded into `no_rel` by
`notebooks/01_prepare_data.ipynb`; pairs using them are excluded from the agreement figures
in `notebooks/02_agreement.ipynb`.
