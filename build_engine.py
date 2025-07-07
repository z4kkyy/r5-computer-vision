import os

import tensorrt as trt


def build_engine_from_onnx(onnx_path, engine_path, fp16=True):
    """Build TensorRT engine from ONNX model"""

    print(f"Building TensorRT engine from {onnx_path}...")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # Memory configuration (4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # FP16 mode configuration (RTX 4060 supports FP16)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")

    # Create network
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)

    # Create ONNX parser
    parser = trt.OnnxParser(network, logger)

    # Load ONNX file
    print("Parsing ONNX file...")
    with open(onnx_path, 'rb') as f:
        onnx_data = f.read()
        if not parser.parse(onnx_data):
            print("ERROR: Failed to parse ONNX file")
            for i in range(parser.num_errors):
                error = parser.get_error(i)
                print(f"Error {i}: {error}")
            return False

    print("ONNX file parsed successfully")

    # Check input shapes
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"Input {i}: {input_tensor.name}, shape: {input_tensor.shape}")

    # Check output shapes
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"Output {i}: {output_tensor.name}, shape: {output_tensor.shape}")

    # Build engine
    print("Building engine... (this may take a few minutes)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return False

    # Save engine
    print(f"Saving engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"Engine saved successfully! Size: {os.path.getsize(engine_path) / 1024 / 1024:.2f} MB")
    return True


if __name__ == "__main__":
    onnx_path = os.path.join("model", "best_8s.onnx")
    engine_path = os.path.join("model", "best_8s.engine")

    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found at {onnx_path}")
        print(f"Current directory: {os.getcwd()}")
        exit(1)

    model_dir = "model"
    if not os.path.exists(model_dir):
        print(f"Creating directory: {model_dir}")
        os.makedirs(model_dir)

    # Build engine
    success = build_engine_from_onnx(onnx_path, engine_path, fp16=True)

    if success:
        print("\nEngine conversion completed successfully!")
        print(f"New engine saved as: {engine_path}")
    else:
        print("\nEngine conversion failed!")
