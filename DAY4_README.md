# Day 4: Advanced Visualization and Reporting

This document provides instructions for running and testing the Day 4 implementation of the Deepfake Detector project, which focuses on advanced visualization and reporting features.

## Features Implemented

1. **Visualization Module**
   - Standard visualizations with annotations
   - Heatmap visualizations showing probability distributions
   - Comparison visualizations combining multiple views

2. **PDF Report Generation**
   - Detailed PDF reports with analysis and recommendations
   - Multiple report formats (PDF, HTML, JSON)
   - Customizable report content

3. **Dashboard for Analytics**
   - Overview of detection statistics
   - Charts and graphs for data visualization
   - Recent detections table with actions

4. **Enhanced API**
   - New endpoints for visualizations and reports
   - Dashboard data endpoint for analytics
   - Improved error handling

## Project Structure

- `visualization.py` - Visualization module for detection results
- `pdf_report.py` - PDF report generation module
- `simple_api.py` - Enhanced API with new endpoints
- `src/pages/Dashboard.tsx` - Dashboard component for the frontend
- `run_day4.bat` - Script to run the Day 4 implementation

## Running the Project

### Prerequisites

Make sure you have the following installed:
- Python 3.8 or higher
- Node.js and npm
- Required Python packages: `flask`, `flask-cors`, `opencv-python`, `numpy`, `matplotlib`, `fpdf`
- Required npm packages (install with `npm install`)

### Option 1: Using the Comprehensive Run Script

1. Run the comprehensive script:
   ```
   .\run_project.bat
   ```

2. Choose option 2 when prompted:
   ```
   Choose which version to run:
   1. Minimal API and Frontend
   2. Simple API and React Frontend
   3. Full Project
   Enter your choice (1-3): 2
   ```

3. Access the application at http://localhost:8080

### Option 2: Using the Day 4 Script

1. Run the Day 4 script:
   ```
   .\run_day4.bat
   ```

2. Access the application at http://localhost:8080

### Option 3: Manual Startup

1. Start the API server:
   ```
   python simple_api.py
   ```

2. In a separate terminal, start the frontend:
   ```
   npm run dev
   ```

3. Access the application at http://localhost:8080

## Troubleshooting

If you encounter issues:

1. **White Screen Issue**
   - Try the minimal frontend first: `.\run_minimal_frontend.bat`
   - Check browser console for errors
   - Verify API server is running: http://localhost:5000/api/models

2. **API Server Issues**
   - Try the minimal API: `.\run_minimal.bat`
   - Check for error messages in the terminal
   - Verify all required packages are installed

3. **Frontend Issues**
   - Clear browser cache
   - Try an incognito/private window
   - Check for JavaScript errors in the console

## Testing

Run the test script to verify everything is working:
```
python test_project.py
```

This will:
- Check the directory structure
- Start the minimal API server
- Test all API endpoints
- Open the minimal frontend

## Next Steps

After successfully testing the Day 4 implementation, you can proceed to Day 5, which will focus on:
- Advanced detection techniques
- Multi-modal analysis
- Enhanced user interface
- Performance optimizations