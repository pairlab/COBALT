package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config holds all the application settings
type Config struct {
	FrameRate int    `json:"frameRate"`
	Codec     string `json:"codec"`
}

// LoadConfig reads the config.json file and parses it into a Config struct
func LoadConfig(filename string) (*Config, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("error opening config file: %w", err)
	}
	defer file.Close()

	var config Config
	if err := json.NewDecoder(file).Decode(&config); err != nil {
		return nil, fmt.Errorf("error decoding config file: %w", err)
	}

	return &config, nil
}
