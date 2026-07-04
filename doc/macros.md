## DEF: Lightweight prompt macros

You can define "prompt macros" by using `DEF`.  Macros are expanded before any other parsing takes place. The expansion continues until no further changes occur. Recursion will raise an error.

`PCLazyTextEncode` and `PCLazyLoraLoader` expand macros, but `PCTextEncode` **does not**. If you need to expand macros for a single prompt, use `PCMacroExpand`

```
DEF(MYMACRO=this is a prompt)
[(MYMACRO:0.6):(MYMACRO:1.1):0.5]
```
is equivalent to
```
[(this is a prompt:0.5):(this is a prompt:1.1):0.5]
```
### Macro parameters
It's also possible to give parameters to a macro:
```
DEF(MYMACRO=[(prompt $1:$2):(prompt $1:$3):$4])
MYMACRO(test; 1.1; 0.7; 0.2)
```
gives
```
[(prompt test:1.1):(prompt test:0.7):0.2]
```
in this form, the variables $N (where N is any number corresponding to a positional parameter) will be replaced with the given parameter. The parameters must be separated with a semicolon, and can be empty.

You can also optionally specify default values:

```
DEF(MACRO(example; 0; 1)=[$1:$2,$3])
MACRO MACRO(test; 0.2)
```
gives
```
[example:0,1] [test:0.2,1]
```

```
DEF(MACRO() = [a:$1:0.5])
```
sets the default value of `$1` to an empty string.

### Unspecified parameters in macros

Unspecified parameters (either via defaults or explicitly given) will not be substituted. Compare:

```
DEF(mything=a "$1" b "$2")
mything
mything()
mything(A)
```

gives

```
a "$1" b "$2"
a "" b "$2"
a "A" b "$2"
```

## SEG: Split your prompt into named segments

Syntax: `SEG(segment_name)`

To help with organizing prompts, you can use the `SEG` function. For example:
```
This is a comic
Top panel: $CAT. $SEG3
Bottom panel: $DOG

SEG(DOG)
A dog chasing its
  tail in a living room.
SEG(CAT)

a sleeping cat

SEG
The cat has orange fur with white stripes
```
This produces:
```
This is a comic
Top panel: a sleeping cat. The cat has orange fur with white stripes
Bottom panel: A dog chasing its
  tail in a living room.
```

> [!NOTE]
> Unlike macros, SEGs are processed *after* scheduling syntax has been expanded, except in the LoRA loader (this may change later, but requires a bit of refactoring)

In this case, the first section before any `SEG` becomes the *template* and any text after a `SEG` call becomes part of that segment. Whitespace is stripped from the start and end of segments and the template.

In the template, you can refer to segments by either their index (starting from 1) or the given name, prefixed with a `$SEG`, so in this example, `$SEG1` is the same as `$PANEL1`

Segments can also refer to each other. Recursion will terminate, but produces weird outputs.

Naming segments is optional, in which case you will have to refer to it by its index.
