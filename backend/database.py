
import sqlite3
import json
import logging
import os
import time
from datetime import datetime
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, Text, select, insert
from backend.config import DB_FILE, DATABASE_URL, USE_SUPABASE

# Configure logging
logger = logging.getLogger(__name__)

# SQLAlchemy setup for Supabase
if USE_SUPABASE:
    try:
        logger.info(f"Attempting to connect to Supabase using DATABASE_URL")
        engine = create_engine(DATABASE_URL)
        metadata = MetaData()
        
        # Define the detection_results table structure
        detection_results = Table(
            'detection_results', 
            metadata,
            Column('id', Integer, primary_key=True),
            Column('image_name', String, nullable=False),
            Column('probability', Float, nullable=False),
            Column('confidence', Float),
            Column('timestamp', String, nullable=False),
            Column('detection_type', String, nullable=False),
            Column('frame_count', Integer),
            Column('processing_time', Integer),
            Column('regions', Text)
        )
        
        # Create table if it doesn't exist
        metadata.create_all(engine)
        logger.info("Connected to Supabase PostgreSQL database")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {str(e)}")
        logger.info("Falling back to SQLite database")
        USE_SUPABASE = False

def init_db():
    """Initialize the database and create tables if they don't exist"""
    if USE_SUPABASE:
        # Already initialized in the global scope
        return
    
    # Fallback to SQLite
    try:
        logger.info(f"Initializing SQLite database: {DB_FILE}")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create detection_results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            probability REAL NOT NULL,
            confidence REAL,
            timestamp TEXT NOT NULL,
            detection_type TEXT NOT NULL,
            frame_count INTEGER,
            processing_time INTEGER,
            regions TEXT
        )
        ''')
        
        conn.commit()
        logger.info("SQLite database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing SQLite database: {str(e)}", exc_info=True)
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def get_db_status():
    """Check database connectivity and return status information"""
    if USE_SUPABASE:
        try:
            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text("SELECT version();")).fetchone()
                version = result[0] if result else "Unknown"
                return {
                    "connected": True,
                    "type": "PostgreSQL (Supabase)",
                    "version": version
                }
        except Exception as e:
            logger.error(f"Supabase connectivity check failed: {str(e)}")
            return {
                "connected": False,
                "type": "PostgreSQL (Supabase)",
                "error": str(e)
            }
    else:
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('SELECT SQLITE_VERSION()')
            version = cursor.fetchone()
            conn.close()
            
            return {
                "connected": True,
                "type": "SQLite",
                "version": version[0] if version else "Unknown"
            }
        except Exception as e:
            logger.error(f"SQLite connectivity check failed: {str(e)}")
            return {
                "connected": False,
                "type": "SQLite",
                "error": str(e)
            }

def save_detection_result(result):
    """Save a detection result to the database"""
    if USE_SUPABASE:
        try:
            # Convert regions to JSON string
            regions_json = json.dumps(result.get('regions', []))
            
            # Prepare the insert statement
            stmt = insert(detection_results).values(
                image_name=result['imageName'],
                probability=result['probability'],
                confidence=result.get('confidence', 0),
                timestamp=result['timestamp'],
                detection_type=result['detectionType'],
                frame_count=result.get('frameCount'),
                processing_time=result.get('processingTime'),
                regions=regions_json
            )
            
            # Execute the statement
            with engine.connect() as conn:
                result_proxy = conn.execute(stmt)
                conn.commit()
                # Get the ID of the inserted row if possible
                result_id = result_proxy.inserted_primary_key[0] if hasattr(result_proxy, 'inserted_primary_key') else None
                
            logger.info(f"Detection result saved to Supabase with ID: {result_id}")
            return result_id
        except Exception as e:
            logger.error(f"Error saving to Supabase: {str(e)}", exc_info=True)
            return None
    else:
        # Fallback to SQLite
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # Convert regions to JSON string
            regions_json = json.dumps(result.get('regions', []))
            
            cursor.execute('''
            INSERT INTO detection_results 
            (image_name, probability, confidence, timestamp, detection_type, 
             frame_count, processing_time, regions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['imageName'],
                result['probability'],
                result.get('confidence', 0),
                result['timestamp'],
                result['detectionType'],
                result.get('frameCount'),
                result.get('processingTime'),
                regions_json
            ))
            
            # Get the ID of the inserted row
            result_id = cursor.lastrowid
            
            conn.commit()
            logger.info(f"Detection result saved to SQLite with ID: {result_id}")
            
            return result_id
        except Exception as e:
            logger.error(f"Error saving to SQLite: {str(e)}", exc_info=True)
            return None
        finally:
            if 'conn' in locals() and conn:
                conn.close()

def get_detection_history(limit=50):
    """Get detection history from the database"""
    if USE_SUPABASE:
        try:
            # Query using SQLAlchemy
            with engine.connect() as conn:
                query = select(detection_results).order_by(detection_results.c.timestamp.desc()).limit(limit)
                result = conn.execute(query)
                rows = result.fetchall()
            
            # Convert rows to dictionaries
            history = []
            for row in rows:
                item = {
                    'id': row.id,
                    'imageName': row.image_name,
                    'probability': row.probability,
                    'confidence': row.confidence,
                    'timestamp': row.timestamp,
                    'detectionType': row.detection_type,
                    'processingTime': row.processing_time
                }
                
                # Add video-specific fields if applicable
                if row.detection_type == 'video' and row.frame_count is not None:
                    item['frameCount'] = row.frame_count
                    
                # Parse regions JSON
                if row.regions:
                    try:
                        item['regions'] = json.loads(row.regions)
                    except json.JSONDecodeError:
                        item['regions'] = []
                
                history.append(item)
            
            logger.info(f"Retrieved {len(history)} detection history records from Supabase")
            return history
        except Exception as e:
            logger.error(f"Error getting history from Supabase: {str(e)}", exc_info=True)
            return []
    else:
        # Fallback to SQLite
        try:
            conn = sqlite3.connect(DB_FILE)
            conn.row_factory = sqlite3.Row  # This enables column access by name
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM detection_results 
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            history = []
            for row in rows:
                item = {
                    'id': row['id'],
                    'imageName': row['image_name'],
                    'probability': row['probability'],
                    'confidence': row['confidence'],
                    'timestamp': row['timestamp'],
                    'detectionType': row['detection_type'],
                    'processingTime': row['processing_time']
                }
                
                # Add video-specific fields if applicable
                if row['detection_type'] == 'video' and row['frame_count'] is not None:
                    item['frameCount'] = row['frame_count']
                    
                # Parse regions JSON
                if row['regions']:
                    try:
                        item['regions'] = json.loads(row['regions'])
                    except json.JSONDecodeError:
                        item['regions'] = []
                
                history.append(item)
            
            logger.info(f"Retrieved {len(history)} detection history records from SQLite")
            return history
        except Exception as e:
            logger.error(f"Error getting history from SQLite: {str(e)}", exc_info=True)
            return []
        finally:
            if 'conn' in locals() and conn:
                conn.close()

# Additional functions can be added based on requirements...
