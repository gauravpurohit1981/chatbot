import base64
import os

def get_background_image_style(image_path_or_data, is_local=False):
    """
    Generate CSS for a background image.
    
    Args:
        image_path_or_data: Path to image file or base64 data URL.
        is_local: Whether the image is a local file path.
        
    Returns:
        str: HTML with CSS for the background image.
    """
    if is_local:
        # For local SVG files, read the content
        if image_path_or_data.endswith('.svg'):
            try:
                with open(image_path_or_data, 'r') as f:
                    svg_content = f.read()
                    # Convert SVG to data URL
                    b64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
                    background_url = f"data:image/svg+xml;base64,{b64}"
            except Exception as e:
                print(f"Error reading SVG file: {str(e)}")
                # Use a fallback color
                return get_fallback_background_style()
        else:
            # Fallback for other local files
            return get_fallback_background_style()
    else:
        # For data URLs
        background_url = image_path_or_data
    
    return f"""
    <style>
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp > header {{
        background-color: rgba(255, 255, 255, 0.8);
    }}
    .stTextInput, .stNumberInput, .stFileUploader, .stSelectbox {{
        background-color: rgba(255, 255, 255, 0.8) !important;
    }}
    .stMarkdown, .block-container {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 10px;
    }}
    </style>
    """

def get_fallback_background_style():
    """
    Generate a fallback background style.
    
    Returns:
        str: HTML with CSS for a fallback background.
    """
    return """
    <style>
    .stApp {
        background: linear-gradient(to right, #8e9eab, #eef2f3);
    }
    </style>
    """
