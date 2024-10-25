# modify

Install using `pip install -e .`

A library for converting PyTorch models _entirely_ to modules, and for easily modifying them (e.g adding LoRA adapters and the like).  The key ideas are:
  * Everything is a `modify.ModifyModule`, and `modify.ModifyModule` is a subtype of `torch.Module`.
  * Values flowing through `modify` networks are either torch tensors, or (possibly nested) tuples of torch tensors.
  * Modules can only be joined using `modify.Sequential` and `modify.Parallel`:
    - `modify.Sequential.forward` passes the input through a sequence of modules.
    - `modify.Parallel.forward` takes a tuple as the input, and passes each element of the tuple through a different module. It is used e.g. to implement residual connections.
  * This makes modifying models easy! There are only two ways to compose modules: `Sequential` and `Parallel` (and they're implemented in basically the same way, they just have different forwards modules).  And you _never_ modify these in-place.  Instead, you just make a new `Sequential` and `Parallel` recursively deciding to copy or modify their submodules.  This very naturally gives you two models with different behaviour but shared parameters.  This is very useful if e.g. you want a model with LoRA adapters, and a model without LoRA adapters to be simultaneously accessible.
  * You can use most PyTorch modules as-is.  There are a bunch more modules corresponding to functions like `torch.sum`.  These are always the obvious capitalization (e.g. `modify.Sum`).
  * The Modules have a bunch of types describing them:
    - The arguments:
      - Unary (one tensor argument)
      - Binary (takes a tuple of two tensors as an argument, e.g. `Add`)
      - Ternary (takes a tuple of three tensors as an argument)
      - Tuple (takes a tuple of tensors as an argument)
    - Whether it has parameters:
      - Param (has parameters)
      - ParamFree (doesn't have parameters)
    - Whether it is a linear function of the inputs:
      - Linear (is a linear function of the inputs)
      - NonLinear (not a linear function of the inputs)
    - Other info about the function:
      - Restructure (something like view or reshape).
      - Elementwise (applies independently to each element of the input, e.g. most activation functions).
      - Vector (applies independently to each vector in the inputs, e.g. softmax).
      - Matrix (applies independently to each matrix, where matrices are in the last two dimensions).
      - Reduction (Reduces across inputs, like sum).
  * It turns out that most normalization layers are a combination of a fixed nonlinear part, and an affine (bias + scale) part.  We have therefore wrapped normalization layers to give `modify.BatchNorm1d`, which doesn't have an affine part, and `modify.ElementwiseAffine`.
  * You shouldn't need to use custom modules.  If you do, they cannot hold submodules.  You should make these submodules of e.g. `modify._Param` (note the underscore).

New modules:
  * `modify.ElementwiseAffine`
  * `modify.Copy(number_of_copies)`
  * `modify.Print`
  * `modify.Breakpoint`
  * `modify.PrintShape`


TODO:
* Self Attention.
* Set up tests with pytest, in a separate folder.
* Set up examples in a separate examples folder.
* Docs.
* Standard ResNet + nanoGPT examples.
* Work out how to load up pre-existing weights, e.g. from Llama3 weights.
* Linear (and Conv etc.) modules that allow you to pass weight_init and bias_init as PyTorch distributions.
