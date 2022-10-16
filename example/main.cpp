#include <consitent_set/consistent_set.hpp>
#include <consitent_set/consistent_avl.hpp>
#include <consitent_set/versioning_avl.hpp>
#include <consitent_set/locked.hpp>
#include <consitent_set/partitioned.hpp>

template <typename container_at>
void check_compilation() {
    using member_t = typename container_at::value_type;

    container_at container;

}

int main() {
    return 0;
}