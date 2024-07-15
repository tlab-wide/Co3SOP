// Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/rpc/ActorId.h"
#include "carla/sensor/data/Array.h"

namespace carla {
namespace sensor {
namespace data {

  class VoxelDetectionEvent : public Array<rpc::ActorId> {
  public:

    explicit VoxelDetectionEvent(RawData &&data)
      : Array<rpc::ActorId>(0u, std::move(data)) {}
  };

} // namespace data
} // namespace sensor
} // namespace carla