"""
Tests for weather API integration.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from quantum_ctl.integration.weather_api import (
    WeatherPredictor, WeatherDataPoint, WeatherForecast, WeatherSource
)


class TestWeatherDataPoint:
    """Test WeatherDataPoint class."""
    
    def test_weather_data_point_creation(self):
        """Test creating weather data point."""
        point = WeatherDataPoint(
            timestamp=time.time(),
            temperature=25.0,
            humidity=60.0,
            pressure=1013.25,
            wind_speed=5.0,
            wind_direction=180.0,
            solar_radiation=800.0,
            cloud_cover=30.0
        )
        
        assert point.temperature == 25.0
        assert point.humidity == 60.0
        assert point.solar_radiation == 800.0


class TestWeatherForecast:
    """Test WeatherForecast class."""
    
    def test_weather_forecast_creation(self):
        """Test creating weather forecast."""
        current = WeatherDataPoint(
            timestamp=time.time(),
            temperature=22.0,
            humidity=55.0,
            pressure=1013.0,
            wind_speed=3.0,
            wind_direction=90.0,
            solar_radiation=600.0,
            cloud_cover=20.0
        )
        
        forecast = WeatherForecast(
            location="Test City",
            latitude=40.7128,
            longitude=-74.0060,
            timezone="UTC",
            current=current,
            source=WeatherSource.OPENWEATHER
        )
        
        assert forecast.location == "Test City"
        assert forecast.latitude == 40.7128
        assert forecast.current.temperature == 22.0


class TestWeatherPredictor:
    """Test WeatherPredictor class."""
    
    @pytest.fixture
    def weather_predictor(self):
        """Create weather predictor for testing."""
        return WeatherPredictor(
            sources=[WeatherSource.OPENWEATHER],
            api_keys={'openweather': 'test_api_key'}
        )
    
    def test_weather_predictor_init(self, weather_predictor):
        """Test weather predictor initialization."""
        assert len(weather_predictor.sources) == 1
        assert weather_predictor.sources[0] == WeatherSource.OPENWEATHER
        assert weather_predictor.api_keys['openweather'] == 'test_api_key'
    
    def test_cache_key_generation(self, weather_predictor):
        """Test cache key generation."""
        key = weather_predictor._cache_key("New York", 24, ["temperature", "humidity"])
        assert "New York" in key
        assert "24" in key
        assert "humidity_temperature" in key  # Sorted features
    
    def test_cache_validity(self, weather_predictor):
        """Test cache validity checking."""
        current_time = time.time()
        
        # Recent data should be valid
        assert weather_predictor._is_cache_valid(current_time - 100)  # 100 seconds ago
        
        # Old data should be invalid
        assert not weather_predictor._is_cache_valid(current_time - 1000)  # 1000 seconds ago
    
    @pytest.mark.asyncio
    async def test_geocode_location(self, weather_predictor):
        """Test location geocoding."""
        # Mock successful geocoding
        with patch.object(weather_predictor, '_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[
                {'lat': 40.7128, 'lon': -74.0060}
            ])
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            lat, lon = await weather_predictor._geocode_location("New York")
            assert lat == 40.7128
            assert lon == -74.0060
    
    @pytest.mark.asyncio
    async def test_geocode_fallback(self, weather_predictor):
        """Test geocoding fallback for known cities."""
        # Mock failed API call
        with patch.object(weather_predictor, '_session') as mock_session:
            mock_session.get.side_effect = Exception("API Error")
            
            lat, lon = await weather_predictor._geocode_location("new york")
            # Should use fallback coordinates
            assert lat == 40.7128
            assert lon == -74.0060
    
    def test_solar_radiation_estimation(self, weather_predictor):
        """Test solar radiation estimation."""
        # Test daytime
        daytime_timestamp = time.mktime(time.strptime("2023-06-15 12:00:00", "%Y-%m-%d %H:%M:%S"))
        solar_day = weather_predictor._estimate_solar_radiation(0, daytime_timestamp, 40.0)  # No clouds
        assert solar_day > 0
        
        # Test nighttime
        nighttime_timestamp = time.mktime(time.strptime("2023-06-15 00:00:00", "%Y-%m-%d %H:%M:%S"))
        solar_night = weather_predictor._estimate_solar_radiation(0, nighttime_timestamp, 40.0)
        assert solar_night == 0
        
        # Test cloud effect
        solar_cloudy = weather_predictor._estimate_solar_radiation(100, daytime_timestamp, 40.0)  # Full clouds
        solar_clear = weather_predictor._estimate_solar_radiation(0, daytime_timestamp, 40.0)  # No clouds
        assert solar_cloudy < solar_clear
    
    @pytest.mark.asyncio
    async def test_fetch_openweather_mock(self, weather_predictor):
        """Test OpenWeatherMap API fetching with mocked response."""
        mock_current_data = {
            'name': 'Test City',
            'sys': {'country': 'US'},
            'dt': int(time.time()),
            'main': {'temp': 25.0, 'humidity': 60, 'pressure': 1013},
            'wind': {'speed': 5.0, 'deg': 180},
            'clouds': {'all': 20},
            'coord': {'lat': 40.7, 'lon': -74.0},
            'timezone': 'UTC'
        }
        
        mock_forecast_data = {
            'city': {'coord': {'lat': 40.7, 'lon': -74.0}},
            'list': [
                {
                    'dt': int(time.time()) + 3600,
                    'main': {'temp': 24.0, 'humidity': 55, 'pressure': 1012},
                    'wind': {'speed': 4.0, 'deg': 170},
                    'clouds': {'all': 30}
                }
            ]
        }
        
        with patch.object(weather_predictor, '_session') as mock_session:
            # Mock current weather response
            mock_current_response = AsyncMock()
            mock_current_response.status = 200
            mock_current_response.json = AsyncMock(return_value=mock_current_data)
            
            # Mock forecast response
            mock_forecast_response = AsyncMock()
            mock_forecast_response.status = 200
            mock_forecast_response.json = AsyncMock(return_value=mock_forecast_data)
            
            # Setup session mock to return appropriate responses
            async def mock_get(url, params=None):
                if 'weather' in url:
                    return mock_current_response
                elif 'forecast' in url:
                    return mock_forecast_response
                else:
                    raise ValueError(f"Unexpected URL: {url}")
            
            mock_session.get = mock_get
            
            # Test the fetch
            forecast = await weather_predictor._fetch_openweather(40.7, -74.0, 24, ["temperature"])
            
            assert forecast is not None
            assert forecast.location == "Test City, US"
            assert forecast.current.temperature == 25.0
            assert len(forecast.hourly) == 1
            assert forecast.hourly[0].temperature == 24.0
    
    def test_ensemble_average(self, weather_predictor):
        """Test ensemble forecasting."""
        # Create mock forecasts
        current1 = WeatherDataPoint(time.time(), 25.0, 60.0, 1013.0, 5.0, 180.0, 800.0, 20.0)
        current2 = WeatherDataPoint(time.time(), 23.0, 65.0, 1012.0, 4.0, 170.0, 750.0, 30.0)
        
        forecast1 = WeatherForecast("Test", 40.7, -74.0, "UTC", current1, hourly=[current1])
        forecast2 = WeatherForecast("Test", 40.7, -74.0, "UTC", current2, hourly=[current2])
        
        averaged = weather_predictor._ensemble_average([forecast1, forecast2], "Test", 40.7, -74.0)
        
        # Check averaged values
        assert averaged.current.temperature == 24.0  # (25 + 23) / 2
        assert averaged.current.humidity == 62.5  # (60 + 65) / 2
        assert len(averaged.hourly) == 1
    
    @pytest.mark.asyncio
    async def test_health_check(self, weather_predictor):
        """Test weather source health check."""
        # Mock successful response
        with patch.object(weather_predictor, '_fetch_from_source') as mock_fetch:
            mock_fetch.return_value = WeatherForecast(
                "Test", 40.7, -74.0, "UTC",
                WeatherDataPoint(time.time(), 25.0, 60.0, 1013.0, 5.0, 180.0, 800.0, 20.0)
            )
            
            health_status = await weather_predictor.health_check()
            
            assert WeatherSource.OPENWEATHER.value in health_status
            assert health_status[WeatherSource.OPENWEATHER.value] is True
    
    @pytest.mark.asyncio
    async def test_get_forecast_with_cache(self, weather_predictor):
        """Test forecast retrieval with caching."""
        # Mock geocoding
        with patch.object(weather_predictor, '_geocode_location') as mock_geocode:
            mock_geocode.return_value = (40.7, -74.0)
            
            # Mock forecast fetching
            mock_forecast = WeatherForecast(
                "Test City", 40.7, -74.0, "UTC",
                WeatherDataPoint(time.time(), 25.0, 60.0, 1013.0, 5.0, 180.0, 800.0, 20.0),
                hourly=[
                    WeatherDataPoint(time.time() + 3600, 24.0, 55.0, 1012.0, 4.0, 170.0, 750.0, 30.0)
                ]
            )
            
            with patch.object(weather_predictor, '_fetch_from_source') as mock_fetch:
                mock_fetch.return_value = mock_forecast
                
                async with weather_predictor:
                    # First call - should fetch
                    forecast1 = await weather_predictor.get_forecast("Test City", 1, ["temperature"])
                    
                    # Second call - should use cache
                    forecast2 = await weather_predictor.get_forecast("Test City", 1, ["temperature"])
                    
                    assert forecast1.location == "Test City"
                    assert forecast2.location == "Test City"
                    
                    # Should have been called only once due to caching
                    assert mock_fetch.call_count == 1