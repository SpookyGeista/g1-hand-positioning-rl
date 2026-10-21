"""
Script to display left palm link position relative to root in Isaac Sim.
Run this script in Isaac Sim's Script Editor.

This script:
1. Finds the robot articulation
2. Finds the left palm/wrist link
3. Calculates position relative to root
4. Displays it in the console and optionally creates a visual marker
"""

# Import omni modules (must be imported first in Isaac Sim)
import omni.usd
from pxr import UsdGeom, Gf
import numpy as np
import sys

# Configuration - HARDCODED for G1 29DOF with Hand (3 fingers)
# Looking specifically for left_hand_palm_link (not wrist links!)
PALM_LINK_NAME = "left_hand_palm_link"  # HARDCODED - G1 29DOF with Hand palm link
# Will auto-detect. Common paths:
# - Training: /World/envs/env_0/Robot (or similar with multiple envs)
# - Single robot: /World/g1_29dof_with_hand_rev_1_0 or /World/g1_29dof_rev_1_0 or /World/g1_23dof_rev_1_0
ROBOT_PRIM_PATH = None

def print_to_terminal(message):
    """Print to both console and terminal."""
    print(message)
    # Force flush to ensure output is visible
    sys.stdout.flush()

def find_robot_in_stage(stage):
    """Auto-detect robot prim in the stage by searching common paths."""
    search_paths = [
        "/World/envs/env_0/Robot",  # Training setup (Isaac Lab with multiple envs)
        "/World/envs/env_0/robot",
        "/World/g1_29dof_with_hand_rev_1_0",  # G1 29DOF with Hand (3 fingers) - PREFERRED for hand positioning
        "/World/g1_29dof_rev_1_0",  # G1 29DOF without hand
        "/World/g1_23dof_rev_1_0",  # G1 23DOF
        "/World/Robot",
        "/World/robot",
    ]
    
    # Try common paths first
    for path in search_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            print(f"[Found] Robot at: {path}")
            return prim
    
    # If not found, search for prims with "robot" or "g1" in name
    print("[Searching] Robot not found at common paths. Searching for prims with 'robot' or 'g1' in name...")
    
    def search_recursive(prim, depth=0, max_depth=5):
        """Recursively search for robot prim."""
        if depth > max_depth:
            return None
        
        prim_name = prim.GetName().lower()
        # Look for robot, g1, or articulation root
        if ("robot" in prim_name or "g1" in prim_name) and prim.IsValid() and depth <= 2:
            print(f"[Found] Possible robot at: {prim.GetPath()}")
            return prim
        
        for child in prim.GetChildren():
            result = search_recursive(child, depth + 1, max_depth)
            if result:
                return result
        return None
    
    # Start search from World
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        robot = search_recursive(world_prim)
        if robot:
            return robot
    
    return None

def find_robot_articulation(prim_path=None):
    """Find the robot articulation in the stage."""
    # Get stage from omni.usd context
    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()
    
    # If path is provided, try it first
    if prim_path:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            return prim, stage
    
    # Auto-detect robot
    prim = find_robot_in_stage(stage)
    if prim:
        return prim, stage
    
    # If still not found, list available prims
    print("\nERROR: Robot not found!")
    print("\nAvailable prims in scene:")
    print("=" * 60)
    
    def list_prims(prim, indent=0, max_depth=3):
        """List prims in the scene."""
        if indent > max_depth:
            return
        prefix = "  " * indent
        print(f"{prefix}- {prim.GetPath()}")
        for child in prim.GetChildren():
            list_prims(child, indent + 1, max_depth)
    
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        list_prims(world_prim, max_depth=4)
    else:
        # List root prims
        for prim in stage.Traverse():
            if prim.GetPath().GetPathElementCount() <= 2:
                print(f"  - {prim.GetPath()}")
    
    return None, None

def find_link_by_name(stage, robot_prim, link_name):
    """Find a specific link by exact name (exclude joints)."""
    def find_link_recursive(prim, target_name):
        """Recursively search for link with exact name (exclude joints)."""
        prim_name = prim.GetName()
        
        # Skip joints - only look for links
        if "joint" in prim_name.lower():
            return None
        
        # Must be exactly the target name (case-sensitive match for URDF)
        if prim_name == target_name:
            return prim
        
        # Check children
        for child in prim.GetChildren():
            result = find_link_recursive(child, target_name)
            if result:
                return result
        
        return None
    
    link_prim = find_link_recursive(robot_prim, link_name)
    
    if link_prim:
        print(f"[OK] Found link: {link_prim.GetPath()}")
        return link_prim
    else:
        print(f"\nERROR: Could not find link with exact name: '{link_name}'")
        print("\nAvailable links (not joints) under robot:")
        print("=" * 60)
        def print_links(prim, indent=0):
            prim_name = prim.GetName().lower()
            # Only print if it's a link (has "link" in name) and not a joint
            if "link" in prim_name and "joint" not in prim_name:
                full_name = prim.GetName()
                print("  " * indent + f"- {prim.GetPath()} (name: '{full_name}')")
            for child in prim.GetChildren():
                print_links(child, indent + 1)
        print_links(robot_prim)
        print("=" * 60)
        return None

def get_world_transform(prim):
    """Get world transform of a prim.
    
    Note: Even though you use URDF, Isaac Sim converts it to USD internally.
    USD requires a TimeCode parameter even for static transforms.
    Usd.TimeCode.Default() gives the current/default frame state.
    """
    from pxr import Usd
    xformable = UsdGeom.Xformable(prim)
    # Default() gives current state - works for both static URDF and animated USD
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return world_transform

def get_position_from_transform(transform):
    """Extract translation from transform matrix."""
    translation = transform.ExtractTranslation()
    return np.array([translation[0], translation[1], translation[2]])

def create_visual_marker(stage, position, name="/Visuals/left_palm_rel_root"):
    """Create a visual marker at the specified position."""
    try:
        # Create Xform prim for marker
        marker_prim = stage.GetPrimAtPath(name)
        if not marker_prim.IsValid():
            marker_prim = stage.DefinePrim(name, "Xform")
        
        # Set transform
        xformable = UsdGeom.Xformable(marker_prim)
        # Clear existing xform ops
        xformable.ClearXformOpOrder()
        # Add translate op
        xform_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, "")
        xform_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        
        # Add a sphere as visual indicator
        sphere_prim_path = f"{name}/sphere"
        sphere_prim = stage.GetPrimAtPath(sphere_prim_path)
        if not sphere_prim.IsValid():
            sphere_prim = UsdGeom.Sphere.Define(stage, sphere_prim_path)
            sphere_prim.CreateRadiusAttr(0.05)  # 5cm radius
            # Make it visible and colorful (yellow)
            material_prim = stage.DefinePrim(f"{name}/material", "Material")
        
        print(f"Created visual marker at {name}")
        return marker_prim
    except Exception as e:
        print(f"Could not create visual marker: {e}")
        return None

def main():
    """Main function to calculate and display left palm position relative to root."""
    print("\n" + "=" * 80)
    print("=" * 80)
    print("Left Palm Position Relative to Root - Isaac Sim Script")
    print("=" * 80)
    print("=" * 80 + "\n")
    
    # Get stage
    try:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            print("ERROR: No USD stage found. Make sure you have a scene loaded.")
            return
        print(f"[OK] Stage found: {stage}")
    except Exception as e:
        print(f"ERROR getting stage: {e}")
        return
    
    # Find robot
    if ROBOT_PRIM_PATH:
        print(f"\n[Searching] Looking for robot at: {ROBOT_PRIM_PATH}")
    else:
        print(f"\n[Searching] Auto-detecting robot in scene...")
    robot_prim, stage = find_robot_articulation(ROBOT_PRIM_PATH)
    if not robot_prim:
        print("\nERROR: Robot not found!")
        print("Please check the available prims listed above and set ROBOT_PRIM_PATH in the script.")
        return
    print(f"[OK] Robot found at: {robot_prim.GetPath()}")
    
    # Find root/base link (usually the first link or one with "base", "root", "torso" in name)
    print(f"\n[Searching] Looking for root link...")
    root_names = ["pelvis", "base", "root", "torso", "body", "base_link"]  # pelvis first for G1
    root_link = find_link_by_name(stage, robot_prim, root_names)
    if not root_link:
        # Use robot prim itself as root
        root_link = robot_prim
        print(f"[INFO] Using robot prim as root: {root_link.GetPath()}")
    else:
        print(f"[OK] Found root link: {root_link.GetPath()}")
    
    # Find left hand palm link (HARDCODED)
    print(f"\n[Searching] Looking for left hand palm link: '{PALM_LINK_NAME}' (HARDCODED)...")
    palm_link = find_link_by_name(stage, robot_prim, PALM_LINK_NAME)
    if not palm_link:
        print(f"\nERROR: Could not find '{PALM_LINK_NAME}'!")
        print("Please check the available links listed above and update PALM_LINK_NAME in the script.")
        return
    print(f"[OK] Found palm link: {palm_link.GetPath()}")
    
    # Get world transforms
    print(f"\n[Calculating] Getting world transforms...")
    try:
        root_transform = get_world_transform(root_link)
        palm_transform = get_world_transform(palm_link)
        print("[OK] Transforms calculated")
    except Exception as e:
        print(f"ERROR calculating transforms: {e}")
        return
    
    # Extract positions
    root_pos = get_position_from_transform(root_transform)
    palm_pos = get_position_from_transform(palm_transform)
    
    # Calculate relative position
    rel_pos = palm_pos - root_pos
    
    # Display results - FORCE OUTPUT TO TERMINAL
    print("\n" + "=" * 80)
    print("=" * 80)
    print("RESULTS - LEFT PALM POSITION RELATIVE TO ROOT:")
    print("=" * 80)
    print("=" * 80)
    sys.stdout.flush()
    
    print(f"Root position (world):     X={root_pos[0]:.6f}, Y={root_pos[1]:.6f}, Z={root_pos[2]:.6f} m")
    print(f"Palm position (world):     X={palm_pos[0]:.6f}, Y={palm_pos[1]:.6f}, Z={palm_pos[2]:.6f} m")
    print("")
    sys.stdout.flush()
    
    print("Palm position RELATIVE TO ROOT:")
    print(f"  X = {rel_pos[0]:.6f} m")
    print(f"  Y = {rel_pos[1]:.6f} m")
    print(f"  Z = {rel_pos[2]:.6f} m")
    print(f"  Distance = {np.linalg.norm(rel_pos):.6f} m")
    sys.stdout.flush()
    
    print("=" * 80)
    print("=" * 80)
    print("")
    sys.stdout.flush()
    
    # Single line format for easy copying
    result_str = f"Palm relative to root: X={rel_pos[0]:.6f} Y={rel_pos[1]:.6f} Z={rel_pos[2]:.6f}"
    print(result_str)
    sys.stdout.flush()
    print("")
    
    # Try to create visual marker (optional)
    try:
        marker_prim = create_visual_marker(stage, palm_pos)
        if marker_prim:
            print("[OK] Visual marker created at palm position.")
    except Exception as e:
        print(f"[INFO] Visual marker not created: {e}")

# Run the script
if __name__ == "__main__":
    main()

