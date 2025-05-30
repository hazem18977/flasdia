#!/usr/bin/env python3
"""
Test script for the new feature importance API endpoints.
"""

import requests
import json

def test_api_endpoints():
    """Test the new API endpoints"""
    base_url = "http://localhost:5000"
    
    print("Testing Feature Importance API Endpoints")
    print("=" * 50)
    
    # Test data
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
    
    # Test 1: General feature importance
    print("1. Testing general feature importance API...")
    try:
        response = requests.get(f"{base_url}/api/feature_importance/random_forest?lang=en&top_n=5")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: {data['total_features']} features returned")
            print("   Top 3 features:")
            for i, feature in enumerate(data['features'][:3]):
                print(f"     {i+1}. {feature['display_name']}: {feature['percentage']}%")
        else:
            print(f"   ❌ Failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print()
    
    # Test 2: Personalized feature importance
    print("2. Testing personalized feature importance API...")
    try:
        payload = {
            "features": test_features,
            "lang": "en",
            "top_n": 5
        }
        response = requests.post(
            f"{base_url}/api/feature_importance/random_forest/personalized",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: {data['total_features']} features returned")
            print("   Top 3 features:")
            for i, feature in enumerate(data['features'][:3]):
                direction = "↑" if feature['direction'] == 'positive' else "↓"
                print(f"     {i+1}. {feature['display_name']}: {feature['percentage']}% {direction}")
        else:
            print(f"   ❌ Failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print()
    
    # Test 3: Error handling
    print("3. Testing error handling...")
    try:
        # Test invalid model
        response = requests.get(f"{base_url}/api/feature_importance/invalid_model")
        if response.status_code == 400:
            print("   ✅ Correctly rejected invalid model")
        else:
            print(f"   ❌ Should have returned 400, got {response.status_code}")
        
        # Test invalid features
        payload = {
            "features": [1, 2, 3],  # Too few features
            "lang": "en"
        }
        response = requests.post(
            f"{base_url}/api/feature_importance/random_forest/personalized",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 400:
            print("   ✅ Correctly rejected invalid feature count")
        else:
            print(f"   ❌ Should have returned 400, got {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

if __name__ == "__main__":
    print("Note: This test requires the Flask app to be running on localhost:5000")
    print("Start the app with: python app.py")
    print()
    
    try:
        test_api_endpoints()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app. Make sure it's running on localhost:5000")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
