-- Production Database Initialization for Quantum HVAC Control
-- This script creates the necessary tables and indexes for production deployment

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Users and Authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash BYTEA NOT NULL,
    salt BYTEA NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'readonly',
    permissions TEXT[] DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_ip INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL
);

-- Buildings
CREATE TABLE IF NOT EXISTS buildings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    building_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    timezone VARCHAR(50),
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Building zones
CREATE TABLE IF NOT EXISTS building_zones (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    building_id UUID REFERENCES buildings(id) ON DELETE CASCADE,
    zone_id VARCHAR(50) NOT NULL,
    name VARCHAR(255),
    area DECIMAL(10, 2),
    volume DECIMAL(10, 2),
    thermal_mass DECIMAL(10, 2),
    max_heating_power DECIMAL(10, 2),
    max_cooling_power DECIMAL(10, 2),
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(building_id, zone_id)
);

-- Building state history
CREATE TABLE IF NOT EXISTS building_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    building_id UUID REFERENCES buildings(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    zone_temperatures JSONB,
    outside_temperature DECIMAL(5, 2),
    humidity DECIMAL(5, 2),
    occupancy JSONB,
    hvac_power JSONB,
    control_setpoints JSONB,
    weather_data JSONB,
    energy_prices JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimization results
CREATE TABLE IF NOT EXISTS optimization_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    building_id UUID REFERENCES buildings(id) ON DELETE CASCADE,
    optimization_id VARCHAR(255) UNIQUE NOT NULL,
    problem_type VARCHAR(50),
    problem_size INTEGER,
    solver_type VARCHAR(50),
    solver_config JSONB,
    input_data JSONB,
    solution JSONB,
    energy DECIMAL(15, 6),
    solve_time DECIMAL(10, 3),
    solver_info JSONB,
    success BOOLEAN,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cached solutions
CREATE TABLE IF NOT EXISTS solution_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    problem_type VARCHAR(50),
    problem_size INTEGER,
    solution JSONB NOT NULL,
    energy DECIMAL(15, 6),
    solve_time DECIMAL(10, 3),
    solver_info JSONB,
    similarity_features JSONB,
    usage_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- System metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50),
    value DECIMAL(15, 6),
    labels JSONB,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alerts and notifications
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id VARCHAR(255) UNIQUE NOT NULL,
    alert_type VARCHAR(100),
    severity VARCHAR(20),
    title VARCHAR(255),
    message TEXT,
    details JSONB,
    source VARCHAR(100),
    building_id UUID REFERENCES buildings(id) ON DELETE SET NULL,
    status VARCHAR(20) DEFAULT 'active',
    acknowledged_by UUID REFERENCES users(id) ON DELETE SET NULL,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Security audit log
CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100),
    severity VARCHAR(20),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(255),
    source_ip INET,
    resource VARCHAR(255),
    action VARCHAR(100),
    result VARCHAR(50),
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- BMS connections and configurations
CREATE TABLE IF NOT EXISTS bms_connections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    building_id UUID REFERENCES buildings(id) ON DELETE CASCADE,
    connection_name VARCHAR(100),
    protocol VARCHAR(50),
    connection_params JSONB,
    point_mappings JSONB,
    status VARCHAR(20) DEFAULT 'inactive',
    last_connected TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Energy consumption data
CREATE TABLE IF NOT EXISTS energy_consumption (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    building_id UUID REFERENCES buildings(id) ON DELETE CASCADE,
    zone_id VARCHAR(50),
    timestamp TIMESTAMP NOT NULL,
    power_consumption_kw DECIMAL(10, 3),
    energy_cost DECIMAL(10, 4),
    carbon_emissions_kg DECIMAL(10, 3),
    efficiency_score DECIMAL(5, 3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Weather forecast data
CREATE TABLE IF NOT EXISTS weather_forecasts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    location VARCHAR(255),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    source VARCHAR(50),
    forecast_timestamp TIMESTAMP,
    data_timestamp TIMESTAMP,
    temperature DECIMAL(5, 2),
    humidity DECIMAL(5, 2),
    pressure DECIMAL(7, 2),
    wind_speed DECIMAL(5, 2),
    wind_direction DECIMAL(5, 1),
    solar_radiation DECIMAL(8, 2),
    cloud_cover DECIMAL(5, 2),
    precipitation DECIMAL(6, 3),
    confidence_score DECIMAL(4, 3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System configuration
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB,
    description TEXT,
    category VARCHAR(50),
    is_encrypted BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_building_states_building_timestamp ON building_states(building_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_optimization_results_building_created ON optimization_results(building_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_solution_cache_expires_at ON solution_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_timestamp ON system_metrics(metric_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_status_created ON alerts(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_security_events_created ON security_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_energy_consumption_building_timestamp ON energy_consumption(building_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_weather_forecasts_location_timestamp ON weather_forecasts(location, forecast_timestamp DESC);

-- Create partial indexes for active records
CREATE INDEX IF NOT EXISTS idx_users_active ON users(username) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(created_at DESC) WHERE status = 'active';

-- Create GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_buildings_config ON buildings USING GIN(config);
CREATE INDEX IF NOT EXISTS idx_optimization_results_solver_config ON optimization_results USING GIN(solver_config);
CREATE INDEX IF NOT EXISTS idx_system_metrics_labels ON system_metrics USING GIN(labels);

-- Create functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_buildings_updated_at BEFORE UPDATE ON buildings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_bms_connections_updated_at BEFORE UPDATE ON bms_connections
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for cleaning up expired data
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS void AS $$
BEGIN
    -- Clean up expired sessions
    DELETE FROM user_sessions WHERE expires_at < CURRENT_TIMESTAMP;
    
    -- Clean up expired cache entries
    DELETE FROM solution_cache WHERE expires_at < CURRENT_TIMESTAMP;
    
    -- Clean up old metrics (older than 90 days)
    DELETE FROM system_metrics WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    -- Clean up old building states (older than 1 year)
    DELETE FROM building_states WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 year';
    
    -- Clean up old weather forecasts (older than 30 days)
    DELETE FROM weather_forecasts WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    -- Clean up resolved alerts (older than 1 year)
    DELETE FROM alerts WHERE status = 'resolved' AND resolved_at < CURRENT_TIMESTAMP - INTERVAL '1 year';
    
    RAISE NOTICE 'Expired data cleanup completed';
END;
$$ LANGUAGE plpgsql;

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, description, category) VALUES
    ('system_version', '"1.0.0"', 'System version', 'system'),
    ('default_optimization_horizon', '24', 'Default optimization horizon in hours', 'optimization'),
    ('max_concurrent_optimizations', '10', 'Maximum concurrent optimization processes', 'performance'),
    ('cache_ttl_hours', '24', 'Default cache TTL in hours', 'caching'),
    ('session_timeout_minutes', '60', 'User session timeout in minutes', 'security'),
    ('max_failed_login_attempts', '5', 'Maximum failed login attempts before lockout', 'security')
ON CONFLICT (config_key) DO NOTHING;

-- Insert default admin user (password: admin123 - CHANGE IN PRODUCTION!)
-- This is a placeholder - generate proper hash in production
INSERT INTO users (username, email, password_hash, salt, role, permissions) VALUES
    ('admin', 'admin@example.com', E'\\x5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', E'\\x5d41402abc4b2a76b9719d911017c592', 'admin', ARRAY['read', 'write', 'delete', 'admin'])
ON CONFLICT (username) DO NOTHING;

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Create read-only user for monitoring
CREATE USER quantum_hvac_monitor WITH PASSWORD 'monitor_password_change_in_production';
GRANT USAGE ON SCHEMA public TO quantum_hvac_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO quantum_hvac_monitor;

COMMIT;