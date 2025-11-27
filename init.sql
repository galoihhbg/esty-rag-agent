-- Database initialization script for Etsy RAG Agent

-- Table to store training examples
CREATE TABLE IF NOT EXISTS training_examples (
    id SERIAL PRIMARY KEY,
    user_input TEXT NOT NULL,
    correct_output JSONB NOT NULL,
    category VARCHAR(100) DEFAULT 'general',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_validated BOOLEAN DEFAULT FALSE
);

-- Table to store configuration fields
CREATE TABLE IF NOT EXISTS config_fields (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    field_type VARCHAR(50) NOT NULL,
    options JSONB,
    is_required BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table to store color palette
CREATE TABLE IF NOT EXISTS colors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    hex_code VARCHAR(7),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table to store prediction logs
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    user_input TEXT NOT NULL,
    config_used JSONB,
    color_list JSONB,
    result JSONB,
    used_examples JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_training_examples_category ON training_examples(category);
CREATE INDEX IF NOT EXISTS idx_training_examples_validated ON training_examples(is_validated);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_created_at ON prediction_logs(created_at);

-- Insert some default colors
INSERT INTO colors (name, hex_code) VALUES
    ('Red', '#FF0000'),
    ('Blue', '#0000FF'),
    ('Green', '#00FF00'),
    ('Yellow', '#FFFF00'),
    ('Orange', '#FFA500'),
    ('Purple', '#800080'),
    ('Pink', '#FFC0CB'),
    ('Black', '#000000'),
    ('White', '#FFFFFF'),
    ('Brown', '#A52A2A'),
    ('Gray', '#808080'),
    ('Navy', '#000080'),
    ('Teal', '#008080'),
    ('Coral', '#FF7F50'),
    ('Gold', '#FFD700')
ON CONFLICT (name) DO NOTHING;
