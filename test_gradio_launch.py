#!/usr/bin/env python3
"""
Test script to diagnose Gradio launch issues
"""

import sys
import traceback

print("="*80)
print("Gradio Launch Test")
print("="*80)

try:
    print("\n1. Testing imports...")
    import gradio as gr
    import pandas as pd
    import numpy as np
    print("   ✅ Basic imports OK")

    print("\n2. Testing simple Gradio app...")
    with gr.Blocks() as simple_demo:
        gr.Markdown("# Test App")
        with gr.Row():
            text_input = gr.Textbox(label="Input")
            text_output = gr.Textbox(label="Output")

        def echo(text):
            return f"Echo: {text}"

        text_input.change(fn=echo, inputs=text_input, outputs=text_output)

    print("   ✅ Simple app created OK")

    print("\n3. Testing import of main app...")
    try:
        # Try importing without running
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gradio_residual_tft_app",
            "gradio_residual_tft_app.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            print("   ⏳ Loading module...")
            spec.loader.exec_module(module)
            print("   ✅ Main app module loaded OK")

            if hasattr(module, 'create_unified_interface'):
                print("   ✅ create_unified_interface function found")
            else:
                print("   ❌ create_unified_interface function NOT found")

    except Exception as e:
        print(f"   ❌ Error importing main app: {e}")
        traceback.print_exc()

    print("\n4. Test launching simple app...")
    print("   If this works, the issue is with the main app code")
    print("   Launching on http://127.0.0.1:7860")
    simple_demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        quiet=False
    )

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
