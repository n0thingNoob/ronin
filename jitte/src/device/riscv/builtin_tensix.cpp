// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_map>
#include <utility>

#include "riscv/builtin_tensix.hpp"

namespace {

#define DECL_BUILTIN(name, count) \
    {TensixBuiltinId::name, {#name, count}},

std::unordered_map<TensixBuiltinId, std::pair<std::string, int>> tensix_builtin_map = {
TENSIX_BUILTINS
};

#undef DECL_BUILTIN

} // namespace

std::unordered_map<TensixBuiltinId, std::pair<std::string, int>> &get_tensix_builtin_map() {
    return tensix_builtin_map;
}
