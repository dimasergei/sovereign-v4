//! Mixture of Experts for Regime-Specialized Confidence Calibration
//!
//! Implements a Mixture of Experts (MoE) architecture with 4 regime-specialized
//! calibrators. Each expert specializes in a specific market regime, and the
//! gating mechanism routes predictions based on detected regime probabilities.
//!
//! Benefits over single calibrator:
//! - Better adaptation to different market conditions
//! - Reduced interference between regime-specific patterns
//! - More stable learning (experts don't forget each other's knowledge)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::info;

use super::learner::{ConfidenceCalibrator, TradeOutcome};
use super::regime::Regime;

/// Number of features in the linear model (must match learner.rs)
const NUM_FEATURES: usize = 6;

/// Expert wrapper around ConfidenceCalibrator
///
/// Each expert specializes in a specific market regime and tracks
/// its own performance statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expert {
    /// The underlying calibrator
    calibrator: ConfidenceCalibrator,
    /// The regime this expert specializes in
    regime: Regime,
    /// Total trades handled by this expert
    trade_count: u32,
    /// Running win rate for this expert
    win_rate: f64,
}

impl Expert {
    /// Create a new expert for a specific regime
    pub fn new(regime: Regime) -> Self {
        Self {
            calibrator: ConfidenceCalibrator::new(),
            regime,
            trade_count: 0,
            win_rate: 0.5, // Start with neutral assumption
        }
    }

    /// Get the expert's regime
    pub fn regime(&self) -> Regime {
        self.regime
    }

    /// Get the expert's trade count
    pub fn trade_count(&self) -> u32 {
        self.trade_count
    }

    /// Get the expert's win rate
    pub fn win_rate(&self) -> f64 {
        self.win_rate
    }

    /// Get a reference to the underlying calibrator
    pub fn calibrator(&self) -> &ConfidenceCalibrator {
        &self.calibrator
    }

    /// Get a mutable reference to the underlying calibrator
    pub fn calibrator_mut(&mut self) -> &mut ConfidenceCalibrator {
        &mut self.calibrator
    }

    /// Predict confidence for given features
    pub fn predict(&self, sr_score: i32, volume_pct: f64) -> f64 {
        // Note: we pass the expert's regime for feature encoding
        self.calibrator.predict(sr_score, volume_pct, &self.regime)
    }

    /// Update the expert with a trade outcome
    pub fn update(&mut self, features: &[f64; NUM_FEATURES], target: f64, learning_rate: f64) {
        self.calibrator.update(features, target, learning_rate);

        // Update statistics
        self.trade_count += 1;
        let won = target > 0.5;
        // Exponential moving average for win rate
        let alpha = 0.1_f64.min(1.0 / self.trade_count as f64);
        self.win_rate = self.win_rate * (1.0 - alpha) + if won { 1.0 } else { 0.0 } * alpha;
    }

    /// Consolidate EWC for this expert
    pub fn consolidate(&mut self, recent_trades: &[TradeOutcome]) {
        self.calibrator.compute_fisher(recent_trades);
        self.calibrator.consolidate();
    }
}

/// Mixture of Experts for regime-specialized confidence calibration
///
/// Routes predictions and updates to regime-specialized experts using
/// a soft gating mechanism based on regime probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixtureOfExperts {
    /// One expert per regime (TrendingUp, TrendingDown, Ranging, Volatile)
    experts: [Expert; 4],
    /// Gating weights (softmax routing probabilities)
    gating_weights: [f64; 4],
    /// Temperature for softmax (lower = sharper routing)
    temperature: f64,
}

impl Default for MixtureOfExperts {
    fn default() -> Self {
        Self::new()
    }
}

impl MixtureOfExperts {
    /// Create a new MoE with one expert per regime
    pub fn new() -> Self {
        let experts = [
            Expert::new(Regime::TrendingUp),
            Expert::new(Regime::TrendingDown),
            Expert::new(Regime::Ranging),
            Expert::new(Regime::Volatile),
        ];

        Self {
            experts,
            gating_weights: [0.25, 0.25, 0.25, 0.25], // Start uniform
            temperature: 1.0,
        }
    }

    /// Predict confidence by blending all expert predictions
    ///
    /// Returns weighted average: sum(gating_i * expert_i.predict())
    pub fn predict(&self, sr_score: i32, volume_pct: f64, regime: &Regime) -> f64 {
        // Encode features once
        let features = ConfidenceCalibrator::encode_features(sr_score, volume_pct, regime);

        // Blend predictions from all experts weighted by gating
        let mut weighted_sum = 0.0;
        for (i, expert) in self.experts.iter().enumerate() {
            let pred = expert.calibrator.predict_from_features(&features);
            weighted_sum += self.gating_weights[i] * pred;
        }

        info!(
            "[MOE] Routing to {} expert (w={:.2}), blended pred={:.3}",
            regime,
            self.gating_weights[regime.index()],
            weighted_sum
        );

        weighted_sum
    }

    /// Get prediction from a specific expert (hard routing)
    pub fn predict_from_expert(&self, sr_score: i32, volume_pct: f64, regime: &Regime) -> f64 {
        let expert = &self.experts[regime.index()];
        expert.calibrator.predict(sr_score, volume_pct, regime)
    }

    /// Update the appropriate expert based on current regime
    ///
    /// Routes the update to the expert that matches the trade's regime.
    pub fn update(
        &mut self,
        sr_score: i32,
        volume_pct: f64,
        regime: &Regime,
        won: bool,
        learning_rate: f64,
    ) {
        let features = ConfidenceCalibrator::encode_features(sr_score, volume_pct, regime);
        let target = if won { 1.0 } else { 0.0 };

        // Route to the expert for this regime
        let expert_idx = regime.index();
        self.experts[expert_idx].update(&features, target, learning_rate);

        info!(
            "[MOE] Updated {} expert: trades={}, win_rate={:.2}%",
            regime,
            self.experts[expert_idx].trade_count(),
            self.experts[expert_idx].win_rate() * 100.0
        );
    }

    /// Update gating weights from regime probabilities
    ///
    /// Uses softmax with temperature to convert regime probabilities
    /// into gating weights.
    pub fn update_gating(&mut self, regime_probs: &[f64; 4]) {
        // Apply temperature scaling and softmax
        let mut exp_probs = [0.0; 4];
        let mut sum = 0.0;

        for i in 0..4 {
            // Temperature scaling: lower temp = sharper routing
            exp_probs[i] = (regime_probs[i] / self.temperature).exp();
            sum += exp_probs[i];
        }

        // Normalize
        if sum > 0.0 {
            for i in 0..4 {
                self.gating_weights[i] = exp_probs[i] / sum;
            }
        }
    }

    /// Get reference to a specific expert
    pub fn get_expert(&self, regime: &Regime) -> &Expert {
        &self.experts[regime.index()]
    }

    /// Get mutable reference to a specific expert
    pub fn get_expert_mut(&mut self, regime: &Regime) -> &mut Expert {
        &mut self.experts[regime.index()]
    }

    /// Consolidate EWC for a specific expert
    ///
    /// Call this when the regime changes to protect the expert's learned weights.
    pub fn consolidate_expert(&mut self, regime: &Regime, recent_trades: &[TradeOutcome]) {
        let expert = &mut self.experts[regime.index()];
        expert.consolidate(recent_trades);
        info!(
            "[MOE] Consolidated {} expert (trades={}, win_rate={:.2}%)",
            regime,
            expert.trade_count(),
            expert.win_rate() * 100.0
        );
    }

    /// Get statistics for all experts
    ///
    /// Returns: Vec<(Regime, trade_count, win_rate)>
    pub fn get_expert_stats(&self) -> Vec<(Regime, u32, f64)> {
        self.experts
            .iter()
            .map(|e| (e.regime(), e.trade_count(), e.win_rate()))
            .collect()
    }

    /// Get current gating weights
    pub fn gating_weights(&self) -> &[f64; 4] {
        &self.gating_weights
    }

    /// Get temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature.max(0.01); // Prevent division by zero
    }

    /// Get total trades across all experts
    pub fn total_trades(&self) -> u32 {
        self.experts.iter().map(|e| e.trade_count()).sum()
    }

    /// Get overall win rate (weighted by trade count)
    pub fn overall_win_rate(&self) -> f64 {
        let total = self.total_trades();
        if total == 0 {
            return 0.5;
        }

        let weighted_sum: f64 = self
            .experts
            .iter()
            .map(|e| e.win_rate() * e.trade_count() as f64)
            .sum();

        weighted_sum / total as f64
    }

    /// Save MoE to JSON file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        info!("[MOE] Saved to {}", path);
        Ok(())
    }

    /// Load MoE from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let moe: Self = serde_json::from_str(&json)?;
        Ok(moe)
    }

    /// Load from file or create new if file doesn't exist
    pub fn load_or_new<P: AsRef<Path>>(path: P) -> Self {
        match Self::load(path) {
            Ok(moe) => moe,
            Err(_) => Self::new(),
        }
    }

    /// Format expert stats for logging
    pub fn format_stats(&self) -> String {
        let parts: Vec<String> = self
            .experts
            .iter()
            .map(|e| {
                format!(
                    "{}(n={}, wr={:.1}%)",
                    e.regime(),
                    e.trade_count(),
                    e.win_rate() * 100.0
                )
            })
            .collect();
        parts.join(" | ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_creation() {
        let expert = Expert::new(Regime::TrendingUp);
        assert_eq!(expert.regime(), Regime::TrendingUp);
        assert_eq!(expert.trade_count(), 0);
        assert!((expert.win_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_moe_creation() {
        let moe = MixtureOfExperts::new();

        // Should have 4 experts
        assert_eq!(moe.experts.len(), 4);

        // Each expert should have correct regime
        assert_eq!(moe.get_expert(&Regime::TrendingUp).regime(), Regime::TrendingUp);
        assert_eq!(moe.get_expert(&Regime::TrendingDown).regime(), Regime::TrendingDown);
        assert_eq!(moe.get_expert(&Regime::Ranging).regime(), Regime::Ranging);
        assert_eq!(moe.get_expert(&Regime::Volatile).regime(), Regime::Volatile);

        // Gating should be uniform
        for w in moe.gating_weights() {
            assert!((w - 0.25).abs() < 0.001);
        }

        // Temperature should be 1.0
        assert!((moe.temperature() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_moe_predict() {
        let moe = MixtureOfExperts::new();

        // Should return a value in [0, 1]
        let pred = moe.predict(0, 80.0, &Regime::TrendingUp);
        assert!(pred >= 0.0 && pred <= 1.0);

        // Different regimes should give different predictions
        let pred_up = moe.predict(0, 80.0, &Regime::TrendingUp);
        let pred_down = moe.predict(0, 80.0, &Regime::TrendingDown);

        // With default weights they may be similar, but should be valid
        assert!(pred_up >= 0.0 && pred_up <= 1.0);
        assert!(pred_down >= 0.0 && pred_down <= 1.0);
    }

    #[test]
    fn test_moe_update() {
        let mut moe = MixtureOfExperts::new();

        // Update trending up expert
        moe.update(0, 85.0, &Regime::TrendingUp, true, 0.1);

        // Only TrendingUp expert should have trades
        assert_eq!(moe.get_expert(&Regime::TrendingUp).trade_count(), 1);
        assert_eq!(moe.get_expert(&Regime::TrendingDown).trade_count(), 0);
        assert_eq!(moe.get_expert(&Regime::Ranging).trade_count(), 0);
        assert_eq!(moe.get_expert(&Regime::Volatile).trade_count(), 0);

        // Update a different expert
        moe.update(-3, 60.0, &Regime::Volatile, false, 0.1);
        assert_eq!(moe.get_expert(&Regime::Volatile).trade_count(), 1);
    }

    #[test]
    fn test_moe_update_gating() {
        let mut moe = MixtureOfExperts::new();

        // Set strong preference for TrendingUp
        let regime_probs = [0.8, 0.1, 0.05, 0.05];
        moe.update_gating(&regime_probs);

        // TrendingUp should have highest gating weight
        let weights = moe.gating_weights();
        assert!(weights[0] > weights[1]);
        assert!(weights[0] > weights[2]);
        assert!(weights[0] > weights[3]);

        // Weights should still sum to 1
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_moe_temperature() {
        let mut moe = MixtureOfExperts::new();

        // Lower temperature = sharper routing
        moe.set_temperature(0.5);
        let regime_probs = [0.6, 0.2, 0.1, 0.1];
        moe.update_gating(&regime_probs);
        let sharp_weights = *moe.gating_weights();

        // Reset and use higher temperature
        moe.set_temperature(2.0);
        moe.update_gating(&regime_probs);
        let soft_weights = *moe.gating_weights();

        // Sharp weights should be more extreme
        assert!(sharp_weights[0] > soft_weights[0]);
    }

    #[test]
    fn test_moe_expert_stats() {
        let mut moe = MixtureOfExperts::new();

        // Add some trades
        moe.update(0, 80.0, &Regime::TrendingUp, true, 0.1);
        moe.update(0, 80.0, &Regime::TrendingUp, true, 0.1);
        moe.update(-5, 50.0, &Regime::Volatile, false, 0.1);

        let stats = moe.get_expert_stats();
        assert_eq!(stats.len(), 4);

        // Find TrendingUp stats
        let up_stats = stats.iter().find(|(r, _, _)| *r == Regime::TrendingUp).unwrap();
        assert_eq!(up_stats.1, 2); // trade_count
        assert!(up_stats.2 > 0.5); // win_rate should be above 0.5

        // Find Volatile stats
        let vol_stats = stats.iter().find(|(r, _, _)| *r == Regime::Volatile).unwrap();
        assert_eq!(vol_stats.1, 1);
    }

    #[test]
    fn test_moe_total_trades() {
        let mut moe = MixtureOfExperts::new();

        assert_eq!(moe.total_trades(), 0);

        moe.update(0, 80.0, &Regime::TrendingUp, true, 0.1);
        moe.update(0, 80.0, &Regime::TrendingDown, false, 0.1);
        moe.update(0, 80.0, &Regime::Ranging, true, 0.1);

        assert_eq!(moe.total_trades(), 3);
    }

    #[test]
    fn test_moe_save_load() {
        let mut moe = MixtureOfExperts::new();

        // Add some trades to modify state
        moe.update(0, 85.0, &Regime::TrendingUp, true, 0.1);
        moe.update(-3, 60.0, &Regime::Volatile, false, 0.1);
        moe.set_temperature(0.8);

        let regime_probs = [0.5, 0.3, 0.1, 0.1];
        moe.update_gating(&regime_probs);

        // Save
        let path = "/tmp/test_moe.json";
        moe.save(path).unwrap();

        // Load
        let loaded = MixtureOfExperts::load(path).unwrap();

        // Verify state
        assert_eq!(loaded.get_expert(&Regime::TrendingUp).trade_count(), 1);
        assert_eq!(loaded.get_expert(&Regime::Volatile).trade_count(), 1);
        assert!((loaded.temperature() - 0.8).abs() < 0.001);

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_moe_load_or_new() {
        // Non-existent file should return new MoE
        let moe = MixtureOfExperts::load_or_new("/tmp/nonexistent_moe_12345.json");
        assert_eq!(moe.total_trades(), 0);
    }

    #[test]
    fn test_moe_format_stats() {
        let moe = MixtureOfExperts::new();
        let formatted = moe.format_stats();

        assert!(formatted.contains("TrendingUp"));
        assert!(formatted.contains("TrendingDown"));
        assert!(formatted.contains("Ranging"));
        assert!(formatted.contains("Volatile"));
    }

    #[test]
    fn test_expert_update_win_rate() {
        let mut expert = Expert::new(Regime::TrendingUp);

        // Simulate winning trades
        for _ in 0..10 {
            let features = ConfidenceCalibrator::encode_features(0, 80.0, &Regime::TrendingUp);
            expert.update(&features, 1.0, 0.1);
        }

        // Win rate should be above initial 0.5
        assert!(expert.win_rate() > 0.5);

        // Simulate losing trades
        for _ in 0..20 {
            let features = ConfidenceCalibrator::encode_features(-5, 40.0, &Regime::TrendingUp);
            expert.update(&features, 0.0, 0.1);
        }

        // Win rate should decrease
        assert!(expert.win_rate() < 0.5);
    }

    #[test]
    fn test_moe_consolidate_expert() {
        let mut moe = MixtureOfExperts::new();

        // Train the TrendingUp expert
        for _ in 0..10 {
            moe.update(0, 85.0, &Regime::TrendingUp, true, 0.1);
        }

        // Create trade outcomes for consolidation
        let trades: Vec<TradeOutcome> = (0..10)
            .map(|_| TradeOutcome {
                sr_score: 0,
                volume_pct: 85.0,
                regime: Regime::TrendingUp,
                won: true,
            })
            .collect();

        // Consolidate
        moe.consolidate_expert(&Regime::TrendingUp, &trades);

        // Expert should now be consolidated
        assert!(moe.get_expert(&Regime::TrendingUp).calibrator().is_consolidated());

        // Other experts should not be consolidated
        assert!(!moe.get_expert(&Regime::TrendingDown).calibrator().is_consolidated());
    }

    #[test]
    fn test_predict_from_expert() {
        let moe = MixtureOfExperts::new();

        // Hard routing should give prediction from specific expert
        let pred = moe.predict_from_expert(0, 80.0, &Regime::TrendingUp);
        assert!(pred >= 0.0 && pred <= 1.0);
    }

    #[test]
    fn test_overall_win_rate() {
        let mut moe = MixtureOfExperts::new();

        // No trades: default 0.5
        assert!((moe.overall_win_rate() - 0.5).abs() < 0.001);

        // Add winning trades to one expert
        for _ in 0..10 {
            moe.update(0, 85.0, &Regime::TrendingUp, true, 0.1);
        }

        // Add losing trades to another
        for _ in 0..10 {
            moe.update(-5, 40.0, &Regime::Volatile, false, 0.1);
        }

        // Overall should be around 0.5 (balanced wins/losses)
        let overall = moe.overall_win_rate();
        assert!(overall > 0.3 && overall < 0.7);
    }
}
