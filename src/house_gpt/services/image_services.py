from fastapi import UploadFile 
import mimetypes

def get_image_type(file: UploadFile, allowed_image_types: set) -> str:
    if file.content_type and file.content_type in allowed_image_types:
        return file.content_type
    guessed_type, _ = mimetypes.guess_type(file.filename or "")
    return guessed_type or file.content_type or ""