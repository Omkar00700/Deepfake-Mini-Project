
# DeepDefend API Documentation

This document provides details about the DeepDefend API endpoints, request formats, and responses.

## Base URL

All API endpoints are relative to: `http://localhost:5000/api/`

## Endpoints

### Status Check

**Endpoint:** `GET /status`

**Description:** Check if the API is running and database is connected.

**Response:**

```json
{
  "status": "online",
  "message": "DeepDefend API is running",
  "version": "1.0.0",
  "database": {
    "connected": true,
    "type": "SQLite",
    "version": "3.36.0"
  }
}
```

### Image Deepfake Detection

**Endpoint:** `POST /detect`

**Description:** Upload an image file to check for deepfakes.

**Request:**
- Form data with key `image` containing the image file.

**Constraints:**
- Maximum file size: 5MB
- Allowed formats: JPG, JPEG, PNG

**Response:**

```json
{
  "success": true,
  "message": "Image processed successfully",
  "result": {
    "imageName": "example.jpg",
    "probability": 0.78,
    "confidence": 0.92,
    "timestamp": "2023-06-01 14:30:45",
    "detectionType": "image",
    "processingTime": 1234,
    "regions": [
      {
        "x": 100,
        "y": 150,
        "width": 200,
        "height": 200,
        "probability": 0.85
      }
    ]
  }
}
```

### Video Deepfake Detection

**Endpoint:** `POST /detect-video`

**Description:** Upload a video file to check for deepfakes.

**Request:**
- Form data with key `video` containing the video file.

**Constraints:**
- Maximum file size: 50MB
- Allowed formats: MP4, MOV, AVI, WEBM

**Response:**

```json
{
  "success": true,
  "message": "Video processed successfully",
  "result": {
    "imageName": "example.mp4",
    "probability": 0.65,
    "confidence": 0.87,
    "timestamp": "2023-06-01 15:45:22",
    "detectionType": "video",
    "frameCount": 24,
    "processingTime": 8765,
    "regions": [
      {
        "x": 120,
        "y": 160,
        "width": 180,
        "height": 180,
        "probability": 0.72
      }
    ]
  }
}
```

### Detection History

**Endpoint:** `GET /history`

**Description:** Get a history of recent detection results.

**Response:**

```json
{
  "success": true,
  "history": [
    {
      "id": 1,
      "imageName": "example1.jpg",
      "probability": 0.78,
      "confidence": 0.92,
      "timestamp": "2023-06-01 14:30:45",
      "detectionType": "image",
      "processingTime": 1234
    },
    {
      "id": 2,
      "imageName": "example2.mp4",
      "probability": 0.65,
      "confidence": 0.87,
      "timestamp": "2023-06-01 15:45:22",
      "detectionType": "video",
      "frameCount": 24,
      "processingTime": 8765
    }
  ]
}
```

## Error Responses

All API endpoints return appropriate HTTP status codes and error messages in case of failures:

```json
{
  "success": false,
  "message": "Error message explaining what went wrong"
}
```

Common status codes:
- 400: Bad Request (invalid input, file too large, etc.)
- 500: Server Error (processing error, database issue, etc.)

## Rate Limiting

The API currently has no rate limiting implemented in the demo version.

