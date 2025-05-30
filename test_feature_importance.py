#!/usr/bin/env python3
"""
Test script for the feature importance calculation system.
This script tests both general model importance and personalized feature contributions.
"""

import sys
import os
import numpy as np

# Add the current directory to the path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_feature_importance_system():
    """Test the feature importance calculation system"""
    print("Testing Feature Importance Calculation System")
    print("=" * 50)
    
    try:
        # Import the necessary components from app.py
        from app import (
            feature_calculator, 
            get_feature_importance, 
            validate_feature_importance_input,
            models,
            EXPECTED_FEATURES
        )
        
        # Test data - sample patient features
        test_features = [
            45.0,    # age
            1,       # hypertension
            0,       # heart_disease
            28.5,    # bmi
            6.2,     # HbA1c_level
            140.0,   # blood_glucose_level
            0,       # gender_Female
            1,       # gender_Male
            0,       # smoking_history_No Info
            0,       # smoking_history_current
            0,       # smoking_history_ever
            1,       # smoking_history_former
            0,       # smoking_history_never
            0        # smoking_history_not current
        ]
        
        print(f"Test features length: {len(test_features)}")
        print(f"Expected features length: {len(EXPECTED_FEATURES)}")
        print(f"Available models: {list(models.keys())}")
        print(f"Loaded models: {[k for k, v in models.items() if v is not None]}")
        print()
        
        # Test each available model
        for model_type in models.keys():
            if models[model_type] is None:
                print(f"⚠️  Skipping {model_type} - model not loaded")
                continue
                
            print(f"Testing {model_type}:")
            print("-" * 30)
            
            # Test 1: Validation
            print("1. Testing input validation...")
            is_valid, error_msg = validate_feature_importance_input(model_type, test_features)
            if is_valid:
                print("   ✅ Validation passed")
            else:
                print(f"   ❌ Validation failed: {error_msg}")
                continue
            
            # Test 2: General importance
            print("2. Testing general importance calculation...")
            try:
                general_importance = get_feature_importance(
                    model_type=model_type,
                    features=None,
                    lang='en',
                    top_n=5,
                    importance_type='general'
                )
                
                if general_importance:
                    print(f"   ✅ General importance calculated: {len(general_importance)} features")
                    print("   Top 3 features:")
                    for i, feature in enumerate(general_importance[:3]):
                        print(f"     {i+1}. {feature['display_name']}: {feature['percentage']}%")
                else:
                    print("   ⚠️  No general importance data returned")
                    
            except Exception as e:
                print(f"   ❌ General importance failed: {str(e)}")
            
            # Test 3: Personalized contributions
            print("3. Testing personalized contributions...")
            try:
                personalized_importance = get_feature_importance(
                    model_type=model_type,
                    features=test_features,
                    lang='en',
                    top_n=5,
                    importance_type='personalized'
                )
                
                if personalized_importance:
                    print(f"   ✅ Personalized importance calculated: {len(personalized_importance)} features")
                    print("   Top 3 features:")
                    for i, feature in enumerate(personalized_importance[:3]):
                        direction = "↑" if feature['direction'] == 'positive' else "↓"
                        print(f"     {i+1}. {feature['display_name']}: {feature['percentage']}% {direction}")
                else:
                    print("   ⚠️  No personalized importance data returned")
                    
            except Exception as e:
                print(f"   ❌ Personalized importance failed: {str(e)}")
            
            print()
        
        # Test 4: Error handling
        print("Testing error handling:")
        print("-" * 30)
        
        # Test with invalid model
        print("1. Testing invalid model...")
        is_valid, error_msg = validate_feature_importance_input("invalid_model")
        if not is_valid:
            print(f"   ✅ Correctly rejected invalid model: {error_msg}")
        else:
            print("   ❌ Should have rejected invalid model")
        
        # Test with wrong number of features
        print("2. Testing wrong number of features...")
        wrong_features = [1, 2, 3]  # Too few features
        is_valid, error_msg = validate_feature_importance_input("random_forest", wrong_features)
        if not is_valid:
            print(f"   ✅ Correctly rejected wrong feature count: {error_msg}")
        else:
            print("   ❌ Should have rejected wrong feature count")
        
        # Test with invalid feature values
        print("3. Testing invalid feature values...")
        invalid_features = test_features.copy()
        invalid_features[0] = float('nan')  # Invalid value
        is_valid, error_msg = validate_feature_importance_input("random_forest", invalid_features)
        if not is_valid:
            print(f"   ✅ Correctly rejected invalid values: {error_msg}")
        else:
            print("   ❌ Should have rejected invalid values")
        
        print()
        print("✅ Feature importance system testing completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure you're running this from the same directory as app.py")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_importance_system()
