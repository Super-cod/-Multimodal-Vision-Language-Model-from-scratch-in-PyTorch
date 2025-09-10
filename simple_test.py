#!/usr/bin/env python3

try:
    print("Testing imports...")
    from modeling_slip import simplevisionconfig
    print("✅ simplevisionconfig imported")
    
    from text_tokenizer import TextTokenizer
    print("✅ TextTokenizer imported")
    
    from vision_encoder import Vision_encoder
    print("✅ Vision_encoder imported")
    
    from multimodal_projector import MultimodalProjector
    print("✅ MultimodalProjector imported")
    
    from processing_paligemma import ImageProcessing
    print("✅ ImageProcessing imported")
    
    from language_model import LanguageModel
    print("✅ LanguageModel imported")
    
    from multimodal_model import MultimodalModel
    print("✅ MultimodalModel imported")
    
    print("✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
