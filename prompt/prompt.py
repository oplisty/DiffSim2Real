Prompt={
    "inversion_prompt":{
        "description":"The prompt to prompt the VLM accurately describes the synthetic video content",
        "prompt":
"""
# Role
You are a specialist in egocentric robot video captioning and embodied action understanding.

# Task
Given a robot first-person video, produce a **detailed dense caption** that describes the full visual content and action progression from the robot's own viewpoint.

# Instructions
Describe the video strictly in temporal order and strictly from the first-person perspective of the robot.  
Focus on the following aspects throughout the caption:

- how the camera view changes over time
- what appears in the environment and where it is located relative to the center of view
- how the robot aligns itself with objects or surfaces
- how the end-effector or gripper enters the frame and moves
- how objects respond to contact, grasping, pushing, wiping, lifting, or placement
- what the robot appears to be trying to achieve at each stage

The caption must preserve **fine-grained action details**, including subtle adjustments, pauses, re-orientation, near-contact motion, failed attempts, repeated corrections, and final task completion cues.

# Perspective Constraint
Always use first-person visual phrasing.  
Prefer expressions such as:
- "the view shifts left"
- "a flat surface comes closer in the center"
- "the gripper enters from the lower edge"
- "the target object becomes aligned with the center of the frame"

Avoid external phrasing such as:
- "the robot walks to"
- "the robot picks up"
- "the robot looks at"

Instead, describe these events as they are visually perceived from the robot's own viewpoint.

# Content Constraint
The caption should include:
- scene layout
- relative spatial relations
- object motion
- manipulator motion
- contact events
- task progression
- likely short-horizon intention

# Output Format
Write the output as several coherent paragraphs in natural language.  
Do not use bullet points, timestamps, or JSON unless explicitly requested.  
Make the caption detailed, specific, and temporally faithful rather than concise.

# Final Instruction
Generate a dense first-perDo not provide a short summary.  
Do not skip intermediate motion details.  
Do not collapse multiple actions into one sentence when the video shows fine-grained sequential behavior.  
Make the caption as detailed as possible while staying visually grounded.
""",

    },

    "positivea_prompt": {
        "description":"The prompt to prompt the VLM to generate more description about the light, texture ,eventually guide Diffusion Model elicit photorealism video,",
        "prompt":
"""
# Role
You are a vision analyst specialized in computer graphics, physically based rendering (PBR), inverse rendering, and embodied robot perception. Your task is **not** to summarize the semantic meaning of the scene, but to extract **render-relevant physical attributes** from a robot egocentric video so that the output can be used as structured conditioning for a diffusion model to generate a more photorealistic scene.

# Input
I will provide a **first-person robot video** captured from an egocentric viewpoint. The video may contain camera motion, motion blur, viewpoint changes, partial occlusion, and varying illumination.

# Core Objective
Analyze the video frame by frame or by keyframes, and extract **fine-grained physical scene information** that is useful for guiding image/video diffusion models toward realism.

Focus on:
- Lighting
- Material and texture
- Surface reflectance properties
- Spatial layout of visually important regions
- Camera characteristics
- Temporal consistency across frames

Avoid high-level semantic interpretation unless it is strictly necessary to distinguish material regions.

# Priority
Your output should be optimized for **downstream generative control**, meaning:
1. descriptions must be **localizable in image space**
2. descriptions must be **physically grounded**
3. descriptions must be **useful as prompt/control conditions for diffusion models**
4. descriptions must separate **geometry cues** from **appearance cues**

# Detailed Instructions

## 1. Lighting Analysis
Infer the scene illumination as if you were estimating parameters for inverse rendering.

Describe:
- overall ambient illumination
- dominant light direction(s)
- softness/hardness of shadows
- indoor vs outdoor style lighting cues
- local highlight behavior
- color cast and color temperature
- exposure and dynamic range
- whether reflections indicate secondary bounce lighting
- whether illumination is stable or changes over time

Do **not** merely say “bright” or “dark”; instead describe illumination in terms useful for rendering and photorealistic generation.

## 2. Material and Texture Analysis
For each visually important region, estimate material properties using **PBR terminology** whenever possible:

- **material class**: metal, plastic, painted surface, rubber, wood, fabric, ceramic, glass, liquid, coated surface, matte composite, etc.
- **albedo/base color**
- **roughness**
- **metallic**
- **specular strength**
- **glossiness**
- **transparency / translucency**
- **normal/detail texture**
- **micro-surface variation**
- **wear, dust, stains, scratches, fingerprints, smudges, wrinkles, fiber structure, grain**
- **wetness or oiliness if visible**
- **anisotropy if visible**
- **reflectance consistency across viewing angles**

Do not only name the object. Instead, describe the **appearance-producing properties** of the region.

## 3. Spatial Localization
Every texture/material/light observation must be tied to a spatial region such as:

- upper left / upper center / upper right
- center left / center / center right
- lower left / lower center / lower right
- foreground / midground / background
- near-field / far-field
- screen-space mask-like description if possible

This is critical because the output will be used for **region-aware diffusion prompting**.

## 4. Temporal Consistency
If the video contains multiple frames, track whether material and lighting cues are:
- stable
- transient
- view-dependent
- occluded in some frames
- only weakly observable

If a property is uncertain, explicitly mark it as:
- `low`
- `medium`
- `high`

Do not hallucinate high-confidence details that are not visually supported.

## 5. Camera and Imaging Artifacts
Estimate:
- approximate field of view
- perspective distortion
- motion blur
- rolling shutter artifacts if visible
- depth-of-field blur
- sensor noise / compression artifacts
- auto exposure shifts
- white balance tendency

These affect how the diffusion model should reproduce realism.

## 6. Semantic Suppression
Avoid ordinary scene captioning.

For example, instead of:
- “there is a table in the center”

prefer:
- “center foreground: horizontally extended hard-surface plane with medium-brown low-saturation albedo, semi-gloss clearcoat-like reflectance, subtle wood grain normal variation, moderate roughness”

Only use object identity when necessary to distinguish material regions.

# Output Format
Return **valid JSON only**. Do not include explanations outside the JSON. If the video is long, summarize by **keyframes** or **stable scene segments**.

Use this schema:

```json
{
  "video_summary": {
    "scene_type": "indoor/outdoor/mixed/uncertain",
    "illumination_stability": "stable/slightly varying/strongly varying",
    "camera_motion": "static/slow ego-motion/moderate ego-motion/fast ego-motion",
    "rendering_use_case_note": "brief note on what appearance factors are most important for realistic diffusion generation"
  },
  "global_illumination": {
    "ambient_light": {
      "description": "overall ambient illumination style",
      "color_cast": "warm/cool/neutral/mixed",
      "intensity": "low/medium/high",
      "dynamic_range": "low/medium/high",
      "confidence": "low/medium/high"
    },
    "dominant_environment_light": {
      "hdri_like_description": "estimate of environment lighting appearance",
      "dominant_direction": "e.g. upper right, frontal, overhead, rear-left",
      "softness": "hard/mixed/soft",
      "stability_over_time": "stable/slightly varying/variable",
      "confidence": "low/medium/high"
    }
  },
  "light_sources": [
    {
      "region": "screen-space or scene-space region",
      "type": "directional/point/area/practical/window/uncertain",
      "direction": "e.g. from upper right at shallow angle",
      "color_temperature": "warm/neutral/cool/mixed",
      "intensity": "low/medium/high",
      "shadow_hardness": "hard/medium/soft",
      "evidence": "highlight/shadow/reflection/exposure cue used for inference",
      "confidence": "low/medium/high"
    }
  ],
  "material_regions": [
    {
      "region_id": "R1",
      "spatial_region": "e.g. lower center foreground",
      "depth_layer": "foreground/midground/background",
      "material_category": "metal/plastic/painted surface/rubber/wood/fabric/ceramic/glass/liquid/composite/uncertain",
      "albedo": {
        "color": "hue + saturation + brightness description",
        "uniformity": "uniform/slightly varied/strongly varied"
      },
      "pbr_properties": {
        "roughness": "low/medium/high",
        "metallic": "low/medium/high",
        "specular": "low/medium/high",
        "glossiness": "low/medium/high",
        "transparency": "none/low/medium/high",
        "translucency": "none/low/medium/high"
      },
      "surface_detail": {
        "normal_detail": "flat/fine texture/coarse texture/geometric relief",
        "micro_texture": "e.g. dust, fabric fibers, brushed lines, scratches, smudges",
        "wear_state": "clean/light wear/heavy wear",
        "wetness": "dry/slightly wet/wet/uncertain"
      },
      "view_dependent_effects": {
        "reflection_strength": "none/weak/moderate/strong",
        "reflection_clarity": "blurred/semi-blurred/sharp",
        "fresnel_like_behavior": "none/subtle/visible/strong",
        "anisotropy": "none/subtle/visible/uncertain"
      },
      "shadow_occlusion": {
        "self_shadowing": "none/weak/moderate/strong",
        "ambient_occlusion_cue": "none/weak/moderate/strong"
      },
      "temporal_consistency": "stable/view-dependent/partially occluded/variable",
      "diffusion_guidance_note": "how this region should be prompted for realistic generation",
      "confidence": "low/medium/high"
    }
  ],
  "camera_metadata": {
    "fov_estimate": "ultra-wide/wide/standard/narrow",
    "perspective_distortion": "low/medium/high",
    "motion_blur": "none/weak/moderate/strong",
    "defocus_blur": "none/weak/moderate/strong",
    "rolling_shutter": "none/weak/possible/strong",
    "exposure_adjustment": "stable/auto-exposure visible/uncertain",
    "white_balance_bias": "warm/cool/neutral/mixed",
    "sensor_or_compression_artifacts": "none/weak/moderate/strong"
  },
  "keyframe_notes": [
    {
      "timestamp_or_segment": "e.g. 0.0-1.5s",
      "appearance_change": "describe any notable lighting/material visibility change",
      "newly_visible_regions": ["R3", "R4"],
      "occluded_regions": ["R1"],
      "confidence": "low/medium/high"
    }
  ],
  "diffusion_rendering_summary": {
    "positive_appearance_prompt": "one concise but detailed prompt describing the physically important appearance cues",
    "negative_appearance_prompt": "artifacts to avoid, such as oversaturated lighting, plastic-looking surfaces, missing contact shadows, inconsistent reflections, texture oversharpening"
  }
}
""",
    }
}