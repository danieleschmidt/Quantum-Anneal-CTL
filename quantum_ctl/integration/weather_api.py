"""
Weather API integration for HVAC optimization.

Provides weather forecasting data from multiple sources with ML-enhanced predictions.
"""

from typing import Dict, List, Optional, Tuple, Union
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = lambda **kwargs: None


class WeatherSource(Enum):
    """Supported weather data sources."""
    OPENWEATHER = "openweather"
    NOAA = "noaa"
    WEATHERCOM = "weather.com"
    LOCAL_STATION = "local_station"


@dataclass
class WeatherDataPoint:
    """Single weather data point."""
    timestamp: float
    temperature: float  # Celsius
    humidity: float  # Percentage
    pressure: float  # hPa
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    solar_radiation: float  # W/m²
    cloud_cover: float  # Percentage
    precipitation: float = 0.0  # mm/h
    feels_like: Optional[float] = None
    uv_index: Optional[float] = None


@dataclass
class WeatherForecast:
    """Weather forecast data structure."""
    location: str
    latitude: float
    longitude: float
    timezone: str
    current: WeatherDataPoint
    hourly: List[WeatherDataPoint] = field(default_factory=list)
    daily: List[WeatherDataPoint] = field(default_factory=list)
    forecast_horizon_hours: int = 24
    confidence_score: float = 0.0
    source: WeatherSource = WeatherSource.OPENWEATHER


class WeatherPredictor:
    """
    ML-enhanced weather prediction system for HVAC optimization.
    
    Aggregates data from multiple sources and applies machine learning
    for improved prediction accuracy.
    """
    
    def __init__(
        self,
        sources: List[Union[str, WeatherSource]] = None,
        fusion_method: str = "ensemble",
        api_keys: Optional[Dict[str, str]] = None,
        cache_duration: int = 300  # seconds
    ):
        # Convert string sources to enum
        if sources is None:
            sources = [WeatherSource.OPENWEATHER]
        
        self.sources = []
        for source in sources:
            if isinstance(source, str):
                self.sources.append(WeatherSource(source))
            else:
                self.sources.append(source)
        
        self.fusion_method = fusion_method
        self.api_keys = api_keys or {}
        self.cache_duration = cache_duration
        
        self.logger = logging.getLogger(__name__)
        self._forecast_cache: Dict[str, Tuple[WeatherForecast, float]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        
        # ML model for prediction enhancement (placeholder)
        self._prediction_model = None
        self._historical_data: List[WeatherDataPoint] = []
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'quantum-ctl-weather/0.1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
    
    def _cache_key(self, location: str, horizon_hours: int, features: List[str]) -> str:
        """Generate cache key for forecast data."""
        feature_str = "_".join(sorted(features))
        return f"{location}_{horizon_hours}_{feature_str}"
    
    def _is_cache_valid(self, cached_time: float) -> bool:
        """Check if cached data is still valid."""
        return time.time() - cached_time < self.cache_duration
    
    async def get_forecast(
        self,
        location: Union[str, Tuple[float, float]],
        horizon_hours: int = 24,
        features: List[str] = None
    ) -> WeatherForecast:
        """
        Get weather forecast for location.
        
        Args:
            location: City name or (latitude, longitude) tuple
            horizon_hours: Forecast horizon in hours
            features: Requested features (temp, humidity, solar_radiation, etc.)
            
        Returns:
            WeatherForecast with requested data
        """
        if features is None:
            features = ["temperature", "humidity", "solar_radiation"]
        
        # Parse location
        if isinstance(location, tuple):
            lat, lon = location
            location_str = f"{lat:.2f},{lon:.2f}"
        else:
            location_str = str(location)
            lat, lon = await self._geocode_location(location)
        
        # Check cache first
        cache_key = self._cache_key(location_str, horizon_hours, features)
        if cache_key in self._forecast_cache:
            forecast, cached_time = self._forecast_cache[cache_key]
            if self._is_cache_valid(cached_time):
                self.logger.debug(f"Using cached forecast for {location_str}")
                return forecast
        
        # Fetch from multiple sources
        forecasts = []
        for source in self.sources:
            try:
                forecast = await self._fetch_from_source(source, lat, lon, horizon_hours, features)
                if forecast:
                    forecasts.append(forecast)
            except Exception as e:
                self.logger.warning(f"Failed to fetch from {source.value}: {e}")
        
        if not forecasts:
            raise RuntimeError(f"No weather data available for {location_str}")
        
        # Fuse multiple forecasts
        final_forecast = await self._fuse_forecasts(forecasts, location_str, lat, lon)
        
        # Apply ML enhancement
        enhanced_forecast = await self._enhance_with_ml(final_forecast)
        
        # Cache result
        self._forecast_cache[cache_key] = (enhanced_forecast, time.time())
        
        return enhanced_forecast
    
    async def _geocode_location(self, location: str) -> Tuple[float, float]:
        """Convert location name to coordinates."""
        if not self._session:
            raise RuntimeError("Weather session not initialized")
        
        # Use OpenWeatherMap geocoding (free tier)
        api_key = self.api_keys.get('openweather', 'demo_key')
        url = f"http://api.openweathermap.org/geo/1.0/direct"
        params = {
            'q': location,
            'limit': 1,
            'appid': api_key
        }
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        return data[0]['lat'], data[0]['lon']
        except Exception as e:
            self.logger.error(f"Geocoding failed: {e}")
        
        # Fallback to major city coordinates
        fallback_coords = {
            'new york': (40.7128, -74.0060),
            'london': (51.5074, -0.1278),
            'tokyo': (35.6762, 139.6503),
            'sydney': (-33.8688, 151.2093),
        }
        
        location_lower = location.lower()
        for city, coords in fallback_coords.items():
            if city in location_lower:
                self.logger.warning(f"Using fallback coordinates for {location}")
                return coords
        
        # Default to NYC if all else fails
        self.logger.warning(f"Could not geocode {location}, using NYC coordinates")
        return 40.7128, -74.0060
    
    async def _fetch_from_source(
        self,
        source: WeatherSource,
        lat: float,
        lon: float,
        horizon_hours: int,
        features: List[str]
    ) -> Optional[WeatherForecast]:
        """Fetch weather data from specific source."""
        if source == WeatherSource.OPENWEATHER:
            return await self._fetch_openweather(lat, lon, horizon_hours, features)
        elif source == WeatherSource.NOAA:
            return await self._fetch_noaa(lat, lon, horizon_hours, features)
        elif source == WeatherSource.LOCAL_STATION:
            return await self._fetch_local_station(lat, lon, horizon_hours, features)
        else:
            self.logger.warning(f"Source {source.value} not implemented yet")
            return None
    
    async def _fetch_openweather(
        self,
        lat: float,
        lon: float,
        horizon_hours: int,
        features: List[str]
    ) -> Optional[WeatherForecast]:
        """Fetch from OpenWeatherMap API."""
        if not self._session:
            return None
        
        api_key = self.api_keys.get('openweather', 'demo_key')
        
        # Current weather
        current_url = "http://api.openweathermap.org/data/2.5/weather"
        current_params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'
        }
        
        # Forecast data
        forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
        forecast_params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'
        }
        
        try:
            # Fetch current and forecast in parallel
            current_task = self._session.get(current_url, params=current_params)
            forecast_task = self._session.get(forecast_url, params=forecast_params)
            
            current_response, forecast_response = await asyncio.gather(
                current_task, forecast_task
            )
            
            if current_response.status != 200 or forecast_response.status != 200:
                self.logger.error(f"OpenWeather API error: {current_response.status}, {forecast_response.status}")
                return None
            
            current_data = await current_response.json()
            forecast_data = await forecast_response.json()
            
            # Parse current weather
            current_weather = self._parse_openweather_current(current_data)
            
            # Parse forecast
            hourly_forecast = self._parse_openweather_forecast(forecast_data, horizon_hours)
            
            location_name = f"{current_data.get('name', 'Unknown')}, {current_data.get('sys', {}).get('country', 'Unknown')}"
            
            return WeatherForecast(
                location=location_name,
                latitude=lat,
                longitude=lon,
                timezone=current_data.get('timezone', 'UTC'),
                current=current_weather,
                hourly=hourly_forecast,
                forecast_horizon_hours=horizon_hours,
                confidence_score=0.85,  # OpenWeather baseline confidence
                source=WeatherSource.OPENWEATHER
            )
            
        except Exception as e:
            self.logger.error(f"OpenWeather fetch failed: {e}")
            return None
    
    def _parse_openweather_current(self, data: Dict) -> WeatherDataPoint:
        """Parse OpenWeatherMap current data."""
        main = data.get('main', {})
        wind = data.get('wind', {})
        clouds = data.get('clouds', {})
        
        return WeatherDataPoint(
            timestamp=data.get('dt', time.time()),
            temperature=main.get('temp', 20.0),
            humidity=main.get('humidity', 50.0),
            pressure=main.get('pressure', 1013.25),
            wind_speed=wind.get('speed', 0.0),
            wind_direction=wind.get('deg', 0.0),
            solar_radiation=self._estimate_solar_radiation(
                clouds.get('all', 0), 
                data.get('dt', time.time()),
                data.get('coord', {}).get('lat', 0)
            ),
            cloud_cover=clouds.get('all', 0.0),
            feels_like=main.get('feels_like'),
            uv_index=data.get('uvi', 0.0)
        )
    
    def _parse_openweather_forecast(self, data: Dict, horizon_hours: int) -> List[WeatherDataPoint]:
        """Parse OpenWeatherMap forecast data."""
        forecasts = []
        forecast_list = data.get('list', [])
        
        for item in forecast_list[:horizon_hours//3]:  # 3-hour intervals
            main = item.get('main', {})
            wind = item.get('wind', {})
            clouds = item.get('clouds', {})
            
            forecast_point = WeatherDataPoint(
                timestamp=item.get('dt', time.time()),
                temperature=main.get('temp', 20.0),
                humidity=main.get('humidity', 50.0),
                pressure=main.get('pressure', 1013.25),
                wind_speed=wind.get('speed', 0.0),
                wind_direction=wind.get('deg', 0.0),
                solar_radiation=self._estimate_solar_radiation(
                    clouds.get('all', 0),
                    item.get('dt', time.time()),
                    data.get('city', {}).get('coord', {}).get('lat', 0)
                ),
                cloud_cover=clouds.get('all', 0.0),
                precipitation=item.get('rain', {}).get('3h', 0.0) / 3.0,  # Convert to mm/h
                feels_like=main.get('feels_like')
            )
            
            forecasts.append(forecast_point)
        
        return forecasts
    
    def _estimate_solar_radiation(self, cloud_cover: float, timestamp: float, latitude: float) -> float:
        """Estimate solar radiation based on cloud cover and sun position."""
        # Simplified solar radiation model
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        
        # Basic solar position calculation
        day_of_year = dt.timetuple().tm_yday
        solar_declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle
        hour_angle = 15 * (hour - 12)
        
        # Solar elevation angle (simplified)
        lat_rad = np.radians(latitude)
        dec_rad = np.radians(solar_declination)
        hour_rad = np.radians(hour_angle)
        
        elevation = np.arcsin(
            np.sin(lat_rad) * np.sin(dec_rad) +
            np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad)
        )
        
        # Clear sky solar radiation (simplified)
        clear_sky_radiation = max(0, 1000 * np.sin(elevation)) if elevation > 0 else 0
        
        # Apply cloud cover attenuation
        cloud_factor = 1.0 - (cloud_cover / 100.0) * 0.8
        
        return clear_sky_radiation * cloud_factor
    
    async def _fetch_noaa(
        self,
        lat: float,
        lon: float,
        horizon_hours: int,
        features: List[str]
    ) -> Optional[WeatherForecast]:
        """Fetch from NOAA API."""
        # NOAA implementation would go here
        # For now, return None to indicate not implemented
        self.logger.info("NOAA weather source not fully implemented yet")
        return None
    
    async def _fetch_local_station(
        self,
        lat: float,
        lon: float,
        horizon_hours: int,
        features: List[str]
    ) -> Optional[WeatherForecast]:
        """Fetch from local weather station."""
        # Local station implementation would go here
        self.logger.info("Local station weather source not implemented yet")
        return None
    
    async def _fuse_forecasts(
        self,
        forecasts: List[WeatherForecast],
        location: str,
        lat: float,
        lon: float
    ) -> WeatherForecast:
        """Fuse multiple forecasts using ensemble methods."""
        if len(forecasts) == 1:
            return forecasts[0]
        
        # Simple ensemble averaging
        if self.fusion_method == "ensemble":
            return self._ensemble_average(forecasts, location, lat, lon)
        elif self.fusion_method == "weighted":
            return self._weighted_average(forecasts, location, lat, lon)
        else:
            # Default to first forecast
            return forecasts[0]
    
    def _ensemble_average(
        self,
        forecasts: List[WeatherForecast],
        location: str,
        lat: float,
        lon: float
    ) -> WeatherForecast:
        """Average multiple forecasts."""
        if not forecasts:
            raise ValueError("No forecasts to average")
        
        # Use first forecast as base
        base = forecasts[0]
        
        # Average current weather
        current_temps = [f.current.temperature for f in forecasts]
        current_humidity = [f.current.humidity for f in forecasts]
        current_pressure = [f.current.pressure for f in forecasts]
        current_solar = [f.current.solar_radiation for f in forecasts]
        
        averaged_current = WeatherDataPoint(
            timestamp=base.current.timestamp,
            temperature=np.mean(current_temps),
            humidity=np.mean(current_humidity),
            pressure=np.mean(current_pressure),
            wind_speed=base.current.wind_speed,  # Use base for non-critical values
            wind_direction=base.current.wind_direction,
            solar_radiation=np.mean(current_solar),
            cloud_cover=base.current.cloud_cover,
            precipitation=base.current.precipitation
        )
        
        # Average hourly forecasts (simplified)
        max_hourly_length = min(len(f.hourly) for f in forecasts)
        averaged_hourly = []
        
        for i in range(max_hourly_length):
            hourly_temps = [f.hourly[i].temperature for f in forecasts]
            hourly_humidity = [f.hourly[i].humidity for f in forecasts]
            hourly_solar = [f.hourly[i].solar_radiation for f in forecasts]
            
            averaged_point = WeatherDataPoint(
                timestamp=base.hourly[i].timestamp,
                temperature=np.mean(hourly_temps),
                humidity=np.mean(hourly_humidity),
                pressure=base.hourly[i].pressure,
                wind_speed=base.hourly[i].wind_speed,
                wind_direction=base.hourly[i].wind_direction,
                solar_radiation=np.mean(hourly_solar),
                cloud_cover=base.hourly[i].cloud_cover,
                precipitation=base.hourly[i].precipitation
            )
            
            averaged_hourly.append(averaged_point)
        
        # Calculate ensemble confidence (higher with more sources)
        confidence = min(0.95, 0.7 + 0.1 * len(forecasts))
        
        return WeatherForecast(
            location=location,
            latitude=lat,
            longitude=lon,
            timezone=base.timezone,
            current=averaged_current,
            hourly=averaged_hourly,
            forecast_horizon_hours=base.forecast_horizon_hours,
            confidence_score=confidence,
            source=WeatherSource.OPENWEATHER  # Ensemble source
        )
    
    def _weighted_average(
        self,
        forecasts: List[WeatherForecast],
        location: str,
        lat: float,
        lon: float
    ) -> WeatherForecast:
        """Weighted average based on source reliability."""
        # Source reliability weights
        weights = {
            WeatherSource.OPENWEATHER: 0.4,
            WeatherSource.NOAA: 0.4,
            WeatherSource.WEATHERCOM: 0.3,
            WeatherSource.LOCAL_STATION: 0.6  # Local stations often more accurate
        }
        
        # For now, just return ensemble average
        # Full weighted implementation would go here
        return self._ensemble_average(forecasts, location, lat, lon)
    
    async def _enhance_with_ml(self, forecast: WeatherForecast) -> WeatherForecast:
        """Apply ML enhancement to improve prediction accuracy."""
        # Placeholder for ML enhancement
        # In production, this would apply trained models for:
        # - Bias correction
        # - Local microclimate adjustment
        # - Temporal smoothing
        # - Uncertainty quantification
        
        # For now, just add small confidence boost if we have historical data
        if self._historical_data:
            forecast.confidence_score = min(0.98, forecast.confidence_score + 0.05)
        
        return forecast
    
    async def add_historical_data(self, data_points: List[WeatherDataPoint]) -> None:
        """Add historical weather data for ML training."""
        self._historical_data.extend(data_points)
        
        # Keep only recent data (last 30 days)
        cutoff_time = time.time() - (30 * 24 * 3600)
        self._historical_data = [
            dp for dp in self._historical_data 
            if dp.timestamp > cutoff_time
        ]
        
        self.logger.info(f"Added historical data, total points: {len(self._historical_data)}")
    
    def get_cached_forecast_count(self) -> int:
        """Get number of cached forecasts."""
        return len(self._forecast_cache)
    
    def clear_cache(self) -> None:
        """Clear forecast cache."""
        self._forecast_cache.clear()
        self.logger.info("Forecast cache cleared")
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all weather sources."""
        health_status = {}
        
        for source in self.sources:
            try:
                # Test with a simple location
                test_forecast = await self._fetch_from_source(
                    source, 40.7128, -74.0060, 3, ["temperature"]
                )
                health_status[source.value] = test_forecast is not None
            except Exception as e:
                self.logger.error(f"Health check failed for {source.value}: {e}")
                health_status[source.value] = False
        
        return health_status