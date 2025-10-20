# Contributing to SmashTable

## Code and Documentation Styling Guidelines

Code is formatted automatically using `clang-format` with the configuration specified in `.clang-format`.
CMake is formatted using `cmake-format` with the configuration specified in `.cmake-format.py`.
Please ensure your code adheres to this style before submitting a pull request.

Documentation is trickier.
All docstrings must use Doxygen-style comments.
Here are a couple of examples.
For minimalistic single-line descriptions:

```cpp
/** @brief Brief one-line description of the function or class. */
```

For more complex functions with many parameters, use the following template:


```cpp
/**
 *  @brief Some function similar to STL's @c map::try_emplace().
 *  @see https://en.cppreference.com/w/cpp/container/map/try_emplace.html
 *  @sa @c insert_if_missing() provides the same functionality under a less ambiguous name.
 *  @param[in] key Object comparable and convertible to @c element_t.
 *  @return pair<iterator, bool> Pair consisting of an iterator to the inserted or existing element.
 *  @retval second True if a new element was inserted, false if an existing element was found.
 */
```

For multi-line descriptions, use 4 spaces for continuation indents:

```cpp
/**
 *  @brief Returns the number of elements with key equal to the specified argument.
 *    For unique-key containers like this, returns either 0 or 1.
 *
 *  @param[in] comparable Object comparable to @c element_t.
 *  @return std::size_t Number of elements with key equal to @p comparable (0 or 1).
 */
```

Use `@see` for external references and `@sa` for internal cross-references.
Use `@retval` for enumerated return values.
Mark class template parameters with `@tparam`.
Mark function parameters with direction indicators: `@param[in]`, `@param[out]`, or `@param[inout]`.
Mention them with `@p`.
Mark class/type names and method names with `@c`.
Use `@b` for bold text and `@a` for italics.
Put examples in `@code` ... `@endcode` blocks.
Keep whitespaces on both sides of tags - `[ @p example]` not `[@p example]`.
