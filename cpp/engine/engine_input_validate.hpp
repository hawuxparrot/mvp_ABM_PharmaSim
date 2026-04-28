#pragma once

#include "engine_input.hpp"
#include "engine_input_view.hpp"

#include <string>

/// Structural validation only (lengths, CSR). Matches Python ``EngineInput.validate_shapes`` intent.
/// Domain rules stay in Python ``validate_scenario``.
[[nodiscard]] std::string validate_engine_input_message(const EngineInput& in) noexcept;

/// @throws std::invalid_argument on failure.
void validate_engine_input_or_throw(const EngineInput& in);

void validate_engine_input_or_throw(const EngineInputView& in);
