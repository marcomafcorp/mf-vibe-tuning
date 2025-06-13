import React from 'react';

const TemperatureDisplay = ({ temperature }) => {
  // Default to 0 if temperature is not provided
  const temp = temperature || 0;
  
  // Determine status icon based on temperature
  const getStatusIcon = (temp) => {
    if (temp < 70) return '🟢'; // Green for cool
    if (temp < 80) return '🟡'; // Yellow for warm
    return '🔴'; // Red for hot
  };

  return (
    <span className="text-sm font-semibold text-orange-400 flex items-center gap-1">
      {temp}°C
      <span className="text-xs">
        {getStatusIcon(temp)}
      </span>
    </span>
  );
};

export default TemperatureDisplay;
