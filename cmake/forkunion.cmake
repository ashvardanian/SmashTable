# The pin for what only the suites and benchmarks include: ForkUnion's atomic references and waiters, which they pass
# into the stores as template arguments. `smashtable::header` never links it, so no consumer inherits it.
# No `FIND_PACKAGE_ARGS`: an installed ForkUnion reports its version with or without the verbs the suites call.
# No `GIT_SHALLOW`: the pin is a commit, which a shallow clone cannot name.
include(FetchContent)
FetchContent_Declare(
    forkunion
    GIT_REPOSITORY https://github.com/ashvardanian/ForkUnion.git
    GIT_TAG 32875b51beffda631acbe4ea24486d01fb4526f3
    EXCLUDE_FROM_ALL SYSTEM
)
FetchContent_MakeAvailable(forkunion)
