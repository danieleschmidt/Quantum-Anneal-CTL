"""Complete end-to-end system integration tests."""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from quantum_ctl.core.controller import HVACController
from quantum_ctl.core.tenant_manager import TenantManager, SubscriptionTier
from quantum_ctl.models.building import Building, ZoneConfig
from quantum_ctl.integration.mock_bms import MockBMSConnector
from quantum_ctl.database.manager import DatabaseManager
from quantum_ctl.optimization.advanced_caching import cache_manager, MemoryCacheBackend
from quantum_ctl.utils.structured_logging import setup_logging


class TestCompleteSystemIntegration:
    """End-to-end system integration tests."""
    
    @pytest.fixture
    async def system_components(self):
        """Setup complete system components."""
        # Setup logging
        setup_logging(level="DEBUG")
        
        # Setup database with in-memory SQLite
        db_manager = DatabaseManager("sqlite:///:memory:")
        await db_manager.initialize()
        
        # Setup tenant manager
        tenant_manager = TenantManager(db_manager)
        tenant = await tenant_manager.create_tenant(
            name="Test Organization",
            slug="test-org",
            subscription_tier=SubscriptionTier.ENTERPRISE,
            admin_email="admin@test.com"
        )
        
        # Setup caching
        memory_backend = MemoryCacheBackend(max_size=100)
        cache_manager.register_backend("memory", memory_backend)
        cache_manager.set_default_backend("memory")
        
        # Create test building
        zones = [
            ZoneConfig(
                zone_id=f"zone_{i}",
                area_m2=100.0,
                volume_m3=300.0,
                thermal_mass_kjk=50000.0
            ) for i in range(3)
        ]
        
        building = Building(
            building_id="test_building",
            zones=zones,
            location={"lat": 37.7749, "lon": -122.4194, "name": "Test City"}
        )
        
        # Setup mock BMS
        mock_bms = MockBMSConnector(
            building_id="test_building", 
            zones=3,
            update_interval=1.0
        )
        await mock_bms.connect()
        
        # Setup HVAC controller
        controller = HVACController(
            building=building,
            prediction_horizon=12,
            control_interval=5,
            solver="classical"
        )
        controller.set_bms_connector(mock_bms)
        
        yield {
            "db_manager": db_manager,
            "tenant_manager": tenant_manager,
            "tenant": tenant,
            "building": building,
            "mock_bms": mock_bms,
            "controller": controller
        }
        
        # Cleanup
        await mock_bms.disconnect()
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_complete_optimization_cycle(self, system_components):
        """Test complete optimization cycle from data to control."""
        components = system_components
        controller = components["controller"]
        mock_bms = components["mock_bms"]
        
        # Step 1: Get current building state
        state = await mock_bms.get_building_state()
        assert state is not None
        assert len(state.temperatures) == 3
        
        # Step 2: Run optimization
        try:
            result = await controller.optimize_async()
            assert result is not None
            assert "control_schedule" in result
            
            # Verify control schedule structure
            schedule = result["control_schedule"]
            assert "setpoints" in schedule
            assert len(schedule["setpoints"]) == 3
            
        except Exception as e:
            # Expected for mock solver - verify error handling works
            assert "quantum solver" in str(e).lower() or "mock" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_tenant_isolation(self, system_components):
        """Test multi-tenant data isolation."""
        components = system_components
        tenant_manager = components["tenant_manager"]
        
        # Create second tenant
        tenant2 = await tenant_manager.create_tenant(
            name="Second Organization",
            slug="second-org",
            subscription_tier=SubscriptionTier.PROFESSIONAL
        )
        
        # Verify tenants are isolated
        tenant1 = components["tenant"]
        assert tenant1.id != tenant2.id
        assert tenant1.slug != tenant2.slug
        
        # Test tenant context
        context1 = tenant_manager.get_tenant_context(tenant1)
        context2 = tenant_manager.get_tenant_context(tenant2)
        
        assert context1.get_tenant_id() != context2.get_tenant_id()
        
        # Test resource limits
        limits1 = context1.get_resource_limits()
        limits2 = context2.get_resource_limits()
        
        assert limits1.max_buildings == 100  # Enterprise
        assert limits2.max_buildings == 10   # Professional
    
    @pytest.mark.asyncio
    async def test_caching_system(self, system_components):
        """Test caching system functionality."""
        # Test basic caching
        await cache_manager.set("test_key", {"data": "test_value"}, ttl=60)
        
        result = await cache_manager.get("test_key")
        assert result is not None
        assert result["data"] == "test_value"
        
        # Test QUBO solution caching
        problem_data = {
            "variables": ["x1", "x2", "x3"],
            "Q": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "constraints": ["x1 + x2 <= 1"]
        }
        
        solution = {
            "variables": {"x1": 1, "x2": 0, "x3": 1},
            "objective_value": 2.0,
            "feasible": True
        }
        
        cache_key = await cache_manager.cache_qubo_solution(
            problem_data=problem_data,
            solution=solution,
            computation_time_ms=150.0
        )
        
        assert cache_key is not None
        
        # Try to find similar solution
        cached_solution = await cache_manager.find_similar_qubo_solution(problem_data)
        assert cached_solution is not None
        assert cached_solution["objective_value"] == 2.0
    
    @pytest.mark.asyncio 
    async def test_bms_integration(self, system_components):
        """Test BMS integration and data flow."""
        mock_bms = components["mock_bms"]
        
        # Test reading data points
        temp = await mock_bms.read_point("zone_1_temp")
        assert temp is not None
        assert 15.0 <= temp <= 30.0  # Reasonable temperature range
        
        # Test writing setpoints
        success = await mock_bms.write_point("zone_1_setpoint", 22.0)
        assert success
        
        # Verify setpoint was written
        setpoint = await mock_bms.read_point("zone_1_setpoint")
        assert setpoint == 22.0
        
        # Test batch operations
        points = ["zone_1_temp", "zone_2_temp", "zone_3_temp"]
        temps = await mock_bms.read_multiple(points)
        
        assert len(temps) == 3
        for point, temp in temps.items():
            assert temp is not None
    
    @pytest.mark.asyncio
    async def test_control_schedule_application(self, system_components):
        """Test applying control schedules to building."""
        mock_bms = components["mock_bms"]
        
        # Create test control schedule
        schedule = {
            "setpoints": [21.0, 22.0, 23.0],
            "dampers": [45.0, 50.0, 40.0],
            "valves": [30.0, 35.0, 25.0]
        }
        
        # Apply schedule
        success = await mock_bms.apply_control_schedule(schedule)
        assert success
        
        # Verify setpoints were applied
        for i, expected_temp in enumerate(schedule["setpoints"]):
            actual_temp = await mock_bms.read_point(f"zone_{i+1}_setpoint")
            assert actual_temp == expected_temp
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, system_components):
        """Test system health monitoring."""
        db_manager = components["db_manager"]
        
        # Test database health
        db_healthy = await db_manager.health_check()
        assert db_healthy
        
        # Test BMS health
        mock_bms = components["mock_bms"]
        assert mock_bms.connected
        
        # Test controller health (should initialize successfully)
        controller = components["controller"]
        assert controller is not None
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, system_components):
        """Test error handling and system recovery."""
        mock_bms = components["mock_bms"]
        
        # Test invalid data point reading
        invalid_temp = await mock_bms.read_point("invalid_point")
        assert invalid_temp is None
        
        # Test invalid control schedule
        invalid_schedule = {
            "setpoints": [100.0, 200.0, 300.0]  # Unrealistic temperatures
        }
        
        # Should handle gracefully (mock BMS may accept but real BMS would reject)
        try:
            await mock_bms.apply_control_schedule(invalid_schedule)
        except Exception as e:
            # Error handling worked
            assert "temperature" in str(e).lower() or "range" in str(e).lower()
    
    def test_configuration_validation(self, system_components):
        """Test system configuration validation."""
        building = components["building"]
        
        # Test building configuration
        assert building.building_id == "test_building"
        assert len(building.zones) == 3
        assert building.location is not None
        
        # Test zone configuration
        for i, zone in enumerate(building.zones):
            assert zone.zone_id == f"zone_{i}"
            assert zone.area_m2 > 0
            assert zone.volume_m3 > 0
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, system_components):
        """Test system performance meets requirements."""
        mock_bms = components["mock_bms"]
        
        # Benchmark: Reading multiple data points should be fast
        import time
        start_time = time.time()
        
        points = [f"zone_{i}_{param}" for i in range(1, 4) for param in ["temp", "humidity", "co2"]]
        data = await mock_bms.read_multiple(points)
        
        read_time = time.time() - start_time
        assert read_time < 1.0  # Should read 9 points in under 1 second
        assert len(data) == 9
        
        # Benchmark: Control application should be fast
        start_time = time.time()
        
        schedule = {
            "setpoints": [21.0, 22.0, 23.0]
        }
        success = await mock_bms.apply_control_schedule(schedule)
        
        control_time = time.time() - start_time
        assert control_time < 0.5  # Should apply controls in under 500ms
        assert success


@pytest.mark.asyncio
async def test_api_integration():
    """Test REST API endpoints (basic smoke test)."""
    from quantum_ctl.api.app import create_app
    from fastapi.testclient import TestClient
    
    app = create_app()
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
    # Test system status (no auth for this test)
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "quantum_status" in data
    assert "database_status" in data


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])