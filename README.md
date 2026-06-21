*This project has been created as part of the 42 curriculum by mben-mer*
# 📞 Call Me Maybe

## Table of Contents

- [Description](#description)
- [Instructions](#instructions)
- [Resources](#resources)
- [Algorithm Explanation](#algorithm-explanation)
- [Design Decisions](#design-decisions)
- [Performance Analysis](#performance-analysis)
- [Challenges Faced](#challenges-faced)
- [Testing Strategy](#testing-strategy)
- [Example Usage](#example-usage)

## Description

The Call Me Maybe project implements function calling for small
language models. Given a natural-language prompt and a set of
available function definitions, the program identifies the function
that best matches the request and generates the corresponding
structured call, including its name and arguments. The
implementation targets Qwen3-0.6B and uses constrained decoding to
guarantee 100% valid JSON output, rather than relying on the
model's raw, unconstrained generation.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) for dependency management

### Setup

Clone the repository, then install dependencies:

```bash
uv sync
```

The `llm_sdk` package must be present in the same directory as `src`.

### Running the program

```bash
uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
```

All three arguments are optional. By default, the program reads
input files from `data/input/` and writes the output to
`data/output/function_calls.json`:

```bash
uv run python -m src
```

To specify custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

### Makefile commands

| Command | Description |
|---|---|
| `make install` | Install project dependencies |
| `make run` | Run the main program |
| `make debug` | Run the program in debug mode |
| `make clean` | Remove temporary files and caches |
| `make lint` | Run flake8 and mypy checks |

## Resources

### References

- 42's project subject (provided documentation on function calling
  and constrained decoding)
- YouTube videos on constrained decoding and token-by-token
  generation in language models

### AI usage

AI assistance (Claude) was used for:

- Theoretical clarification and confirmation of logical reasoning
  around constrained decoding and token masking strategies
- Validation of implementation ideas before coding them
- Writing and correcting docstrings (English phrasing and PEP 257
  formatting)
- Drafting and refining this README

All core algorithmic decisions, debugging, and code implementation
were done independently.

## Algorithm Explanation

### Initial setup

The main prompt is first encoded to provide context to the model
(available functions and the user's request). The resulting tokens
are stored as the initial `input_ids`, which grows throughout the
generation process as new tokens are added.

### The 4-step state machine
```
[Open Structure] → [Function Name] → [Open Parameters] → [Arguments] → done
     (fixed)          (constrained)        (fixed)         (constrained)
```

The constrained decoding process is divided into four steps:

1. **Open structure** — writes the fixed JSON prefix
   (`{"prompt": "...", "name": "`)
2. **Function name** — generated token by token under constraint
3. **Open parameters** — writes the fixed transition
   (`", "parameters": {`)
4. **Arguments** — generated token by token under constraint,
   according to each argument's type

Steps 1 and 3 produce fixed structural tokens: they are simply
encoded and appended directly to `input_ids`, since their content
never changes. Steps 2 and 4 are where the model's output is
actually constrained.

### Token masking

For each constrained generation step, all logits are first set to
negative infinity using a NumPy array, which is much faster than
looping individually over roughly 150,000 vocabulary tokens to mask
them one by one.

A list of valid token IDs is then built using the `vocabulary.json`
file (provided by the SDK). The original mapping was inverted
(string → ID became ID → string), which makes it possible to use
`str.startswith()` to identify which tokens are valid candidates at
each step. The real logits of these valid tokens are restored in
the masked array, and `argmax` is used to select the highest-scoring
valid token. The selected token is then appended to both `input_ids`
(to continue guiding generation) and the final output structure.

### Two-prompt approach

Rather than using a single prompt for the whole task, the pipeline
uses two separate prompts: one dedicated to selecting the function
name, and another for extracting the argument values. Separating
these concerns helps the small model focus on one task at a time,
which improves the reliability of both the function choice and the
argument extraction compared to a single combined prompt.

The available functions are also presented to the model as a
simplified, readable summary (each function's name and description
on its own lines) rather than as raw JSON. This reduces the amount
of structural token noise (`{`, `}`, `"`, `:`) the model has to
parse, helping it focus on the semantic match between the request
and each function's description.

### Per-type generation strategy

- **String**: tokens are generated one by one until a non-escaped
  closing quote is produced. Escaped quotes (`\"`) and braces (`{`,
  `}`) are allowed as legitimate string content, since string
  termination is determined solely by the unescaped closing quote.
  A safety limit caps generation length to prevent runaway
  repetition loops.
- **Number / integer**: tokens are generated one by one as long as
  they remain valid digits.
- **Boolean**: rather than generating freely, the token representing
  `true` or `false` is identified directly in `vocabulary.json` and
  forced.
- **None**: handled the same way as booleans, by forcing the token
  corresponding to `null`.
- **Function name**: tokens are constrained to those that are valid
  prefixes of at least one candidate function name. After each
  token is generated, the matched prefix is trimmed from all
  candidate names, progressively narrowing the list until a single
  function name remains.

The number of arguments and their expected types are extracted from
the function definition during step 3, which is what allows step 4
to know exactly which arguments to generate and under which type
constraint.

## Design Decisions

### Raw dictionaries over validated Pydantic objects

Function definitions and prompts are validated against Pydantic
models (`FunctionModel`, `PromptModel`) purely as a safety gate.
After validation, the raw dictionaries are kept and used throughout
the pipeline instead of the validated objects themselves. This
avoids unnecessary conversions between Pydantic models and plain
dictionaries when building the JSON output, while still guaranteeing
that malformed input is caught early.

### Fail-fast on function definitions, skip-and-continue on prompts

A distinction is made between the two input files based on their
role:

- If `functions_definition.json` is missing, invalid, or contains a
  malformed function, the program stops immediately. Function
  definitions are the structural foundation of the entire pipeline;
  without a reliable schema, no valid output can be produced.
- If an individual prompt in `function_calling_tests.json` is
  invalid, it is skipped with a warning, and processing continues
  with the remaining prompts. Prompts are independent of one
  another, so a single malformed entry should not prevent the
  others from being processed.

### Formatted function summary instead of raw JSON

When presenting the available functions to the model for selection,
the definitions are converted into a readable text summary (name
and description per function) rather than passed as raw JSON. The
`returns` field is also omitted, as it is irrelevant to the
selection task. This keeps the context concise and focused on what
matters for choosing the right function.

### Enum-based state machine

The four steps of the generation loop (`OPEN_STRUCTURE`,
`FUNCTION_NAME`, `OPEN_PARAMETERS`, `ARGUMENTS`) are represented
using a Python `Enum` rather than raw integers. This makes the
control flow self-documenting and avoids the use of unexplained
"magic numbers" when comparing or reading the current step.

### Output directory created automatically, input files validated strictly

The parent directory of the output file is created automatically
if it does not exist, to avoid unnecessary friction for the user.
Input files, on the other hand, are strictly validated: missing or
unreadable files cause the program to stop with a clear error
message, since input data cannot be invented or assumed.

### Handling functions with no parameters

Functions with an empty parameters object are handled as a special
case: the empty `{}` is emitted directly instead of entering the
argument-generation loop, which would otherwise attempt to access a
non-existent argument. This ensures functions that take no arguments
produce valid output without crashing.

## Performance Analysis

### Speed

Output generation for a batch of ten prompts completes in well
under 5 minutes on standard hardware, comfortably within the
requirement set by the subject.

### Reliability

Overall reliability is good for typical prompts and stays within
expected accuracy. Known edge cases (leading zeros, negative
numbers, long accented strings) are detailed in "Challenges Faced"
below. In all cases, the constrained decoding mechanism guarantees
that the JSON output remains structurally valid, even when a
generated value is semantically incorrect.

### Prompts with no matching function

If no function definition reasonably matches the prompt, the
program defaults to the first function in the list, while still
correctly generating arguments of the right types for that
function. For vague or ambiguous prompts, the model attempts to
select the function it considers the closest match before falling
back to this default.

## Challenges Faced

### Understanding how to implement constrained decoding

At the start of the project, the main difficulty was not
understanding the project's goal, but figuring out a concrete way
to implement constrained decoding. The subject's explanation and
the available tools (notably `vocabulary.json`) were read several
times before settling on a working approach: using `str.startswith()`
to check whether a token's string representation is a valid
continuation of the expected target (a function name, a string
suffix, a numeric prefix, etc.). Once this idea was found, the rest
of the constrained decoding logic became straightforward, since
every type of constraint (function names, strings, numbers,
booleans) could be expressed using the same prefix-matching
principle.

### Invalid leading zeros in numbers

JSON does not allow numbers with leading zeros directly followed by
another digit (e.g. `007` is invalid). Since digits are generated
one token at a time without semantic awareness of this rule,
generation is stopped as soon as more than one leading zero is
detected, and the value defaults to `0` or `0.0`. This guarantees
the output remains valid JSON, at the cost of not preserving
legitimate decimal values starting with zero (such as `0.07`).

### No reliable support for negative numbers

While the `-` token is not excluded from the valid token set when
generating `number` or `integer` values, its natural logit is
typically too low compared to digit tokens. As a result, `argmax`
selection almost always favors a digit over the minus sign, even
when the prompt implies a negative value. This is a limitation of
the underlying model's confidence on this specific token rather
than a deliberate constraint in the decoding logic.

### Escaped quotes and structural characters inside strings

Generating string values raised two related issues. First, escaped
quotes (`\"`) requested as content could be mistaken for the
string's closing delimiter, since the decoded token ends with `"`.
Second, the model could "escape" out of a string by emitting a
closing brace mid-value, producing invalid JSON.

The solution relies on a single deterministic rule: a string ends
only when a token ends with a quote that is not escaped. With this
rule in place, escaped quotes and braces can be safely allowed as
legitimate string content without breaking JSON validity, since
termination no longer depends on excluding these characters but on
detecting the unescaped closing quote.

### Long strings with accented characters

Testing revealed that the model struggles to reliably reverse long
strings (roughly beyond 20-30 characters) when combined with
non-ASCII characters such as accented letters. In these cases, the
model sometimes generates the literal type name (e.g. `"string"`)
instead of the expected value, rather than producing an incorrect
but plausible string. This is a limitation of the small-scale
model's ability to track exact character sequences over long
spans, not a flaw in the constrained decoding mechanism — the
output JSON remains valid in every observed case.


### Runaway repetition on degenerate prompts

On degenerate prompts (e.g. asking to reverse a lone quote
character), the small model can fall into a repetition loop,
generating the same token endlessly without ever producing a
terminator. A maximum token count per string was added as a safety
limit: once reached, the string is force-closed with a quote. This
guarantees termination and valid JSON even in the worst case, at
the cost of a semantically truncated value for these rare,
non-meaningful prompts.

### Generation speed

Early in development, even the fixed structural tokens (braces,
keys, separators) were being generated token by token through the
constrained decoding loop, going through the logits-masking process
unnecessarily. Switching to directly encoding these fixed segments
and appending them to `input_ids` — skipping the masking and model
inference step entirely for content that never changes — noticeably
reduced total generation time.

## Testing Strategy

The implementation was validated through three categories of tests:

### Provided test cases

The implementation was first validated against the example
`functions_definition.json` and `function_calling_tests.json` files
provided with the subject, covering the baseline expected behavior
(addition, greeting, string reversal, replace with regex, square).

### Edge case testing

Additional prompts were manually crafted to test the limits of the
system, including: empty strings, large numbers, special and
accented characters, ambiguous prompts with no clear matching
function, and functions with multiple parameters of mixed types.
These tests revealed the known limitations documented in
"Challenges Faced" and "Performance Analysis" (leading zeros,
negative numbers, long accented strings).

### File and input error handling

The error-handling paths were tested explicitly by simulating
real-world failure conditions: missing input files, invalid JSON
syntax (trailing commas, malformed structures), empty function
definitions, empty prompt lists, and output paths with insufficient
permissions or non-existent parent directories. Each case was
verified to produce a clear error message and a controlled exit,
without an unhandled exception or program crash.


## Example Usage

### Input
`data/input/functions_definition.json`:
```json
[
    {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": {
        "type": "string"
      },
      "regex": {
        "type": "string"
      },
      "replacement": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  },
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  }
]
```
`data/input/function_calling_tests.json`:
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  { "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"}
]
```

### Running the program

```bash
uv run python -m src
```

### Output

`data/output/function_calls.json`:
```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    },
    {
        "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "The cat sat on the mat with another cat",
            "regex": "cat",
            "replacement": "dog"
        }
    }
]
```

The program prints the processing time for each prompt and the
total elapsed time:

```
Prompt: "What is the sum of 2 and 3?"
Elapsed: 4.32 seconds.
---------
Prompt: "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"
Elapsed: 14.54 seconds.
---------
Total time: 0.31 min.
```