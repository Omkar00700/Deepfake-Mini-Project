import React from 'react';
import { Link } from 'react-router-dom';

const SimpleDashboard = () => {
  return (
    <div style={{ 
      padding: '20px', 
      maxWidth: '800px', 
      margin: '0 auto', 
      fontFamily: 'Arial, sans-serif' 
    }}>
      <h1>Simple Dashboard</h1>
      <p>This is a simplified dashboard to test if the routing is working correctly.</p>
      
      <div style={{ marginTop: '20px' }}>
        <Link to="/" style={{ 
          padding: '10px 15px', 
          backgroundColor: '#4CAF50', 
          color: 'white', 
          textDecoration: 'none', 
          borderRadius: '4px',
          marginRight: '10px'
        }}>
          Home
        </Link>
        
        <Link to="/about" style={{ 
          padding: '10px 15px', 
          backgroundColor: '#2196F3', 
          color: 'white', 
          textDecoration: 'none', 
          borderRadius: '4px' 
        }}>
          About
        </Link>
      </div>
    </div>
  );
};

export default SimpleDashboard;