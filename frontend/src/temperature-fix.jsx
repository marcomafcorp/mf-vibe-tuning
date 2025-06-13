{systemResources?.temperature || 0}°C
<span className="text-xs">
  {(systemResources?.temperature || 0) < 70 ? '🟢' : (systemResources?.temperature || 0) < 80 ? '🟡' : '🔴'}
</span>
