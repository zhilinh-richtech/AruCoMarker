#!/usr/bin/env python3
"""
Extract camera intrinsics directly from Orbbec Gemini 355 SDK.

This script connects to the 355 Gemini camera and extracts both RGB and depth
camera intrinsics using the Orbbec SDK.
"""

import json
import os
import numpy as np
from datetime import datetime
from pyorbbecsdk import Pipeline, Context, Config, OBFormat, OBSensorType, OBStreamType

class GeminiIntrinsicsExtractor:
    """Extract camera intrinsics from Orbbec Gemini 355 camera."""
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.rgb_intrinsics = None
        self.depth_intrinsics = None
        
    def initialize_camera(self):
        """Initialize the camera pipeline."""
        try:
            self.pipeline = Pipeline()
            device_list = Context().query_devices()
            
            if len(device_list) == 0:
                print("❌ No Orbbec devices found!")
                return False
                
            print(f"✓ Found {len(device_list)} Orbbec device(s)")
            
            # Get device info
            device = device_list[0]
            print(f"  Device: {device.get_device_info().get_name()}")
            print(f"  Serial: {device.get_device_info().get_serial_number()}")
            
            # Get stream profiles
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            
            print(f"  Color profiles: {color_profiles.get_count()}")
            print(f"  Depth profiles: {depth_profiles.get_count()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize camera: {e}")
            return False
    
    def extract_rgb_intrinsics(self):
        """Extract RGB camera intrinsics."""
        try:
            print("\n📷 Extracting RGB Camera Intrinsics...")
            
            # Get camera parameters
            camera_param = self.pipeline.get_camera_param()
            if camera_param is None:
                print("❌ Failed to get camera parameters!")
                return None
                
            # Get RGB intrinsics
            rgb_intrinsics = camera_param.rgb_intrinsic
            if rgb_intrinsics is None:
                print("❌ Failed to get RGB intrinsics!")
                return None
                
            # Get distortion coefficients from separate distortion object
            distortion_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
            try:
                rgb_distortion = camera_param.rgb_distortion
                if rgb_distortion is not None:
                    distortion_coeffs = [
                        float(rgb_distortion.k1),
                        float(rgb_distortion.k2),
                        float(rgb_distortion.p1),
                        float(rgb_distortion.p2),
                        float(rgb_distortion.k3)
                    ]
                    print(f"✓ RGB distortion coefficients: {distortion_coeffs}")
                else:
                    print("⚠️ No RGB distortion data available")
            except Exception as e:
                print(f"⚠️ Could not get RGB distortion: {e}")
                pass  # Use default distortion coefficients
                
            self.rgb_intrinsics = {
                'fx': float(rgb_intrinsics.fx),
                'fy': float(rgb_intrinsics.fy),
                'cx': float(rgb_intrinsics.cx),
                'cy': float(rgb_intrinsics.cy),
                'width': int(rgb_intrinsics.width),
                'height': int(rgb_intrinsics.height),
                'distortion': distortion_coeffs,
                'description': f"Orbbec Gemini 355 RGB intrinsics - Auto-extracted ({rgb_intrinsics.width}x{rgb_intrinsics.height})"
            }
            
            print("✓ RGB Intrinsics extracted:")
            print(f"  fx: {self.rgb_intrinsics['fx']:.6f}")
            print(f"  fy: {self.rgb_intrinsics['fy']:.6f}")
            print(f"  cx: {self.rgb_intrinsics['cx']:.6f}")
            print(f"  cy: {self.rgb_intrinsics['cy']:.6f}")
            print(f"  Resolution: {self.rgb_intrinsics['width']}x{self.rgb_intrinsics['height']}")
            print(f"  Distortion: {self.rgb_intrinsics['distortion']}")
            
            return self.rgb_intrinsics
            
        except Exception as e:
            print(f"❌ Failed to extract RGB intrinsics: {e}")
            return None
    
    def extract_depth_intrinsics(self):
        """Extract depth camera intrinsics."""
        try:
            print("\n📏 Extracting Depth Camera Intrinsics...")
            
            # Get camera parameters
            camera_param = self.pipeline.get_camera_param()
            if camera_param is None:
                print("❌ Failed to get camera parameters!")
                return None
                
            # Get depth intrinsics
            depth_intrinsics = camera_param.depth_intrinsic
            if depth_intrinsics is None:
                print("❌ Failed to get depth intrinsics!")
                return None
                
            # Get distortion coefficients from separate distortion object
            distortion_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
            try:
                depth_distortion = camera_param.depth_distortion
                if depth_distortion is not None:
                    distortion_coeffs = [
                        float(depth_distortion.k1),
                        float(depth_distortion.k2),
                        float(depth_distortion.p1),
                        float(depth_distortion.p2),
                        float(depth_distortion.k3)
                    ]
                    print(f"✓ Depth distortion coefficients: {distortion_coeffs}")
                else:
                    print("⚠️ No depth distortion data available")
            except Exception as e:
                print(f"⚠️ Could not get depth distortion: {e}")
                pass  # Use default distortion coefficients
                
            self.depth_intrinsics = {
                'fx': float(depth_intrinsics.fx),
                'fy': float(depth_intrinsics.fy),
                'cx': float(depth_intrinsics.cx),
                'cy': float(depth_intrinsics.cy),
                'width': int(depth_intrinsics.width),
                'height': int(depth_intrinsics.height),
                'distortion': distortion_coeffs,
                'description': f"Orbbec Gemini 355 Depth intrinsics - Auto-extracted ({depth_intrinsics.width}x{depth_intrinsics.height})"
            }
            
            print("✓ Depth Intrinsics extracted:")
            print(f"  fx: {self.depth_intrinsics['fx']:.6f}")
            print(f"  fy: {self.depth_intrinsics['fy']:.6f}")
            print(f"  cx: {self.depth_intrinsics['cx']:.6f}")
            print(f"  cy: {self.depth_intrinsics['cy']:.6f}")
            print(f"  Resolution: {self.depth_intrinsics['width']}x{self.depth_intrinsics['height']}")
            print(f"  Distortion: {self.depth_intrinsics['distortion']}")
            
            return self.depth_intrinsics
            
        except Exception as e:
            print(f"❌ Failed to extract depth intrinsics: {e}")
            return None
    
    def start_streams(self):
        """Start RGB and depth streams to get intrinsics for 1920x1080."""
        try:
            print("\n🎬 Starting camera streams for 1920x1080...")
            
            self.config = Config()
            
            # Enable color stream - look for 1920x1080 profile
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = None
            
            print(f"  Available color profiles: {color_profiles.get_count()}")
            available_resolutions = []
            for i in range(color_profiles.get_count()):
                profile = color_profiles.get_stream_profile_by_index(i)
                if profile.is_video_stream_profile():
                    vp = profile.as_video_stream_profile()
                    resolution = f"{vp.get_width()}x{vp.get_height()}"
                    available_resolutions.append(resolution)
                    print(f"    Profile {i}: {resolution} {vp.get_format()} @ {vp.get_fps()}fps")
                    
                    # Look for 1920x1080 profile
                    if vp.get_width() == 1920 and vp.get_height() == 1080:
                        color_profile = profile
                        print(f"    ✓ Found 1920x1080 color profile")
                        break
            
            # If still no profile found, use first available
            if color_profile is None:
                print("⚠️ No 1920x1080 color profile found, using first available")
                color_profile = color_profiles.get_stream_profile_by_index(0)
                if color_profile.is_video_stream_profile():
                    vp = color_profile.as_video_stream_profile()
                    print(f"    Using: {vp.get_width()}x{vp.get_height()} {vp.get_format()}")
            
            print(f"  Available resolutions: {', '.join(set(available_resolutions))}")
            
            self.config.enable_stream(color_profile)
            print("✓ Color stream enabled")
            
            # Enable depth stream - look for 1280x800 (max available)
            depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = None
            
            print(f"  Available depth profiles: {depth_profiles.get_count()}")
            for i in range(depth_profiles.get_count()):
                profile = depth_profiles.get_stream_profile_by_index(i)
                if profile.is_video_stream_profile():
                    vp = profile.as_video_stream_profile()
                    print(f"    Profile {i}: {vp.get_width()}x{vp.get_height()} {vp.get_format()} @ {vp.get_fps()}fps")
                    
                    # Look for 1280x800 depth profile (maximum available)
                    if vp.get_width() == 1280 and vp.get_height() == 800:
                        depth_profile = profile
                        print(f"    ✓ Found 1280x800 depth profile (maximum resolution)")
                        break
            
            if depth_profile is None:
                print("⚠️ No 1280x800 depth profile found, using first available")
                depth_profile = depth_profiles.get_stream_profile_by_index(0)
                if depth_profile.is_video_stream_profile():
                    vp = depth_profile.as_video_stream_profile()
                    print(f"    Using: {vp.get_width()}x{vp.get_height()} {vp.get_format()}")
            
            self.config.enable_stream(depth_profile)
            print("✓ Depth stream enabled")
            
            # Start pipeline
            self.pipeline.start(self.config)
            print("✓ Pipeline started")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start streams: {e}")
            return False
    
    def save_intrinsics(self, output_dir="gemini_intrinsics"):
        """Save extracted intrinsics to JSON files."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save RGB intrinsics
        if self.rgb_intrinsics:
            rgb_file = os.path.join(output_dir, f"gemini_355_rgb_intrinsics_{timestamp}.json")
            with open(rgb_file, 'w') as f:
                json.dump({
                    "orbbec_gemini_355_rgb": self.rgb_intrinsics
                }, f, indent=2)
            print(f"✓ RGB intrinsics saved to: {rgb_file}")
            K = np.array([
                [self.rgb_intrinsics['fx'], 0, self.rgb_intrinsics['cx']],
                [0, self.rgb_intrinsics['fy'], self.rgb_intrinsics['cy']],
                [0, 0, 1]
            ])

            # Distortion coefficients (numpy array)
            dist = np.array(self.rgb_intrinsics['distortion'])

            # Save intrinsics to .npz
            np.savez("../output/orbbec_calibration.npz",
                    camera_matrix=K,
                    dist_coeffs=dist)
        # Save depth intrinsics
        if self.depth_intrinsics:
            depth_file = os.path.join(output_dir, f"gemini_355_depth_intrinsics_{timestamp}.json")
            with open(depth_file, 'w') as f:
                json.dump({
                    "orbbec_gemini_355_depth": self.depth_intrinsics
                }, f, indent=2)
            print(f"✓ Depth intrinsics saved to: {depth_file}")
        
        # Save combined intrinsics
        if self.rgb_intrinsics or self.depth_intrinsics:
            combined_file = os.path.join(output_dir, f"gemini_355_combined_intrinsics_{timestamp}.json")
            combined_data = {}
            if self.rgb_intrinsics:
                combined_data["orbbec_gemini_355_rgb"] = self.rgb_intrinsics
            if self.depth_intrinsics:
                combined_data["orbbec_gemini_355_depth"] = self.depth_intrinsics
            
            with open(combined_file, 'w') as f:
                json.dump(combined_data, f, indent=2)
            print(f"✓ Combined intrinsics saved to: {combined_file}")
    
    def run(self):
        """Main extraction process for 1920x1080 RGB and 1280x800 depth (maximum available)."""
        print("🎥 Orbbec Gemini 355 Intrinsics Extractor")
        print("  RGB: 1920x1080")
        print("  Depth: 1280x800 (maximum available)")
        print("=" * 60)
        
        # Initialize camera
        if not self.initialize_camera():
            return False
        
        # Start streams
        if not self.start_streams():
            return False
        
        try:
            # Extract RGB intrinsics
            rgb_success = self.extract_rgb_intrinsics() is not None
            
            # Extract depth intrinsics
            depth_success = self.extract_depth_intrinsics() is not None
            
            if rgb_success or depth_success:
                # Save intrinsics
                self.save_intrinsics()
                
                print("\n✅ Intrinsics extraction completed!")
                if rgb_success:
                    print("  ✓ RGB intrinsics extracted")
                if depth_success:
                    print("  ✓ Depth intrinsics extracted")
            else:
                print("\n❌ Failed to extract any intrinsics!")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️ Extraction interrupted by user")
            
        finally:
            # Clean up
            if self.pipeline:
                self.pipeline.stop()
                print("✓ Camera stopped")
        
        return True

def main():
    """Main function."""
    extractor = GeminiIntrinsicsExtractor()
    success = extractor.run()
    
    if success:
        print("\n🎯 Next steps:")
        print("  1. Check the generated JSON files in 'gemini_intrinsics/' directory")
        print("  2. Use these intrinsics in your YOLO pose detection script")
        print("  3. RGB intrinsics will be for 1920x1080 resolution")
        print("  4. Depth intrinsics will be for 1280x800 resolution (maximum available)")
        print("  5. Update the camera intrinsics in your detection code")
    else:
        print("\n❌ Intrinsics extraction failed!")
        print("  Make sure the 355 Gemini camera is connected and accessible")
        print("  Make sure the camera supports 1920x1080 RGB and 1280x800 depth")

if __name__ == "__main__":
    main()
