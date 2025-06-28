# How to Run DeepDefend

This guide provides step-by-step instructions to run the DeepDefend project.

## Step 1: Fix Import Issues

First, run the script to fix import issues in the backend files:

```
cd d:/TusharFinal/deepfake-detector-india-main
python fix_imports.py
```

## Step 2: Run the Backend

You have two options for running the backend:

### Option 1: Try the original backend (may have issues)

```
cd d:/TusharFinal/deepfake-detector-india-main
python run_backend.py
```

### Option 2: Run the simplified backend (recommended for quick demo)

```
cd d:/TusharFinal/deepfake-detector-india-main
python run_simplified_backend.py
```

The backend will start on http://localhost:5000

## Step 3: Run the Frontend

Open a new terminal window and run:

```
cd d:/TusharFinal/deepfake-detector-india-main
npm install
npm run dev
```

The frontend will start and be accessible at http://localhost:5173 or similar (check the terminal output for the exact URL).

## Troubleshooting

If you encounter issues:

1. Make sure both backend and frontend are running in separate terminal windows
2. Check the terminal output for error messages
3. Try the simplified backend if the original backend has issues
4. Ensure all required Python packages are installed:
   ```
   pip install flask flask-cors python-dotenv pyjwt
   ```
5. If the frontend fails to start, try:
   ```
   npm install --force
   ```

## Notes

- The simplified backend provides mock implementations of the API endpoints
- For a full-featured experience, you would need to fix all import issues in the original backend
- The frontend is configured to connect to http://localhost:5000/api