#!/usr/bin/env python3
import argparse
from pathlib import Path

from pxr import Sdf, Usd


ASSET_ROOT = Path("/isaacsim_assets/Assets/Isaac/5.1/Isaac")
WRITABLE_CONTROLLER_ROOT = Path("/isaac-sim/.local/share/ov/data/forklift_controller/Isaac")
DEFAULT_INPUT = ASSET_ROOT / "Samples/ROS2/Robots/forklift_b_ROS.usd"
DEFAULT_OUTPUT = WRITABLE_CONTROLLER_ROOT / "Samples/ROS2/Robots/forklift_b_ROS_controller.usd"
REFERENCE_DIRS = ("Robots", "Sensors", "Materials")


def attr_type_like(prim, attr_name, fallback):
    attr = prim.GetAttribute(attr_name)
    return attr.GetTypeName() if attr and attr.IsValid() else fallback


def create_attr(prim, name, type_name, value=None):
    attr = prim.GetAttribute(name)
    if attr and attr.IsValid() and attr.GetTypeName() != type_name:
        prim.RemoveProperty(name)
        attr = None
    if not attr or not attr.IsValid():
        attr = prim.CreateAttribute(name, type_name, custom=True)
    if value is not None:
        attr.Set(value)
    return attr


def ensure_asset_reference_links(output_path):
    try:
        output_path.resolve().relative_to(WRITABLE_CONTROLLER_ROOT)
    except ValueError:
        return

    WRITABLE_CONTROLLER_ROOT.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_DIRS:
        source = ASSET_ROOT / name
        target = WRITABLE_CONTROLLER_ROOT / name
        if target.exists() or target.is_symlink() or not source.exists():
            continue
        target.symlink_to(source, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    parser.add_argument("--graph", default="/forklift_sensors/drive")
    parser.add_argument("--topic", default="/forklift/lift_cmd")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.input)
    if stage is None:
        raise RuntimeError(f"failed to open {args.input}")

    graph = stage.GetPrimAtPath(args.graph)
    if not graph:
        raise RuntimeError(f"graph not found: {args.graph}")

    ack_sub = stage.GetPrimAtPath(f"{args.graph}/ros2_subscribe_ackermanndrive")
    make_array = stage.GetPrimAtPath(f"{args.graph}/make_array")
    if not ack_sub:
        raise RuntimeError("existing ROS2 Ackermann subscriber was not found")
    if not make_array:
        raise RuntimeError("existing lift/steering ConstructArray node was not found")

    lift_sub = stage.DefinePrim(f"{args.graph}/ros2_subscribe_lift", "OmniGraphNode")
    create_attr(lift_sub, "node:type", Sdf.ValueTypeNames.Token, "isaacsim.ros2.bridge.ROS2Subscriber")
    create_attr(lift_sub, "node:typeVersion", Sdf.ValueTypeNames.Int, 1)
    create_attr(lift_sub, "inputs:context", attr_type_like(ack_sub, "inputs:context", Sdf.ValueTypeNames.UInt64))
    create_attr(lift_sub, "inputs:execIn", attr_type_like(ack_sub, "inputs:execIn", Sdf.ValueTypeNames.Token))
    create_attr(lift_sub, "inputs:messagePackage", Sdf.ValueTypeNames.String, "std_msgs")
    create_attr(lift_sub, "inputs:messageSubfolder", Sdf.ValueTypeNames.String, "msg")
    create_attr(lift_sub, "inputs:messageName", Sdf.ValueTypeNames.String, "Float64")
    create_attr(lift_sub, "inputs:topicName", Sdf.ValueTypeNames.String, args.topic)
    create_attr(lift_sub, "inputs:queueSize", Sdf.ValueTypeNames.UInt64, 10)
    create_attr(lift_sub, "outputs:data", Sdf.ValueTypeNames.Double, 0.0)

    lift_sub.GetAttribute("inputs:context").ClearConnections()
    lift_sub.GetAttribute("inputs:context").AddConnection(Sdf.Path(f"{args.graph}/ros2_context.outputs:context"))
    lift_sub.GetAttribute("inputs:execIn").ClearConnections()
    lift_sub.GetAttribute("inputs:execIn").AddConnection(Sdf.Path(f"{args.graph}/on_playback_tick.outputs:tick"))

    lift_input = make_array.GetAttribute("inputs:input1")
    if not lift_input or not lift_input.IsValid():
        lift_input = make_array.CreateAttribute("inputs:input1", Sdf.ValueTypeNames.Double, custom=True)
    lift_input.ClearConnections()
    lift_input.AddConnection(Sdf.Path(f"{args.graph}/ros2_subscribe_lift.outputs:data"))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ensure_asset_reference_links(out)
    if not stage.GetRootLayer().Export(str(out)):
        raise RuntimeError(f"failed to export patched USD to {out}")

    print(f"wrote {out}")
    print(f"lift topic: {args.topic} std_msgs/msg/Float64")
    print(f"connection: {args.graph}/ros2_subscribe_lift.outputs:data -> {args.graph}/make_array.inputs:input1")


if __name__ == "__main__":
    main()
