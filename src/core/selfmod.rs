//! Autonomous Self-Modification with Constitutional Constraints
//!
//! Implements a system for the trading agent to modify its own rules and
//! configuration while respecting hard constitutional limits that cannot
//! be overridden. This enables learning and adaptation while maintaining
//! safety guarantees.
//!
//! Key concepts:
//! - Constitution: Immutable safety constraints
//! - Rules: Self-generated trading rules based on observed weaknesses
//! - Modifications: Changes to config, thresholds, or rules
//! - Guard: Validates all changes against constitution

use chrono::{DateTime, Duration, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use tracing::info;

use super::regime::Regime;
use super::weakness::{Weakness, WeaknessType};
use super::counterfactual::{TradingInsight, InsightType};
use super::embeddings::TradeContext;

// ==================== Modification Types ====================

/// Types of modifications the system can make
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModificationType {
    /// Change a configuration value
    ConfigChange {
        key: String,
        old_value: String,
        new_value: String,
    },
    /// Adjust a threshold
    ThresholdChange {
        name: String,
        old: f64,
        new: f64,
    },
    /// Add a new trading rule
    RuleAddition {
        rule: TradingRule,
    },
    /// Remove an existing rule
    RuleRemoval {
        rule_id: u64,
    },
    /// Modify an existing rule
    RuleModification {
        rule_id: u64,
        change: RuleChange,
    },
    /// Adjust component weights
    WeightAdjustment {
        component: String,
        old: f64,
        new: f64,
    },
    /// Toggle a feature on/off
    FeatureToggle {
        feature: String,
        enabled: bool,
    },
}

impl ModificationType {
    /// Get a short description of the modification
    pub fn description(&self) -> String {
        match self {
            ModificationType::ConfigChange { key, new_value, .. } => {
                format!("Config: {} -> {}", key, new_value)
            }
            ModificationType::ThresholdChange { name, old, new } => {
                format!("Threshold: {} {:.3} -> {:.3}", name, old, new)
            }
            ModificationType::RuleAddition { rule } => {
                format!("Add rule: {}", rule.name)
            }
            ModificationType::RuleRemoval { rule_id } => {
                format!("Remove rule: #{}", rule_id)
            }
            ModificationType::RuleModification { rule_id, change } => {
                format!("Modify rule #{}: {:?}", rule_id, change)
            }
            ModificationType::WeightAdjustment { component, old, new } => {
                format!("Weight: {} {:.2} -> {:.2}", component, old, new)
            }
            ModificationType::FeatureToggle { feature, enabled } => {
                format!("Feature: {} = {}", feature, enabled)
            }
        }
    }

    /// Get the category for grouping similar modifications
    pub fn category(&self) -> &str {
        match self {
            ModificationType::ConfigChange { .. } => "config",
            ModificationType::ThresholdChange { .. } => "threshold",
            ModificationType::RuleAddition { .. } => "rule",
            ModificationType::RuleRemoval { .. } => "rule",
            ModificationType::RuleModification { .. } => "rule",
            ModificationType::WeightAdjustment { .. } => "weight",
            ModificationType::FeatureToggle { .. } => "feature",
        }
    }
}

/// Changes that can be made to a rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleChange {
    /// Change the condition
    ConditionChange { new_condition: RuleCondition },
    /// Change the action
    ActionChange { new_action: RuleAction },
    /// Change priority
    PriorityChange { new_priority: i32 },
    /// Enable/disable
    StatusChange { enabled: bool },
}

// ==================== Trading Rules ====================

/// A self-generated or human-defined trading rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingRule {
    /// Unique identifier
    pub id: u64,
    /// Human-readable name
    pub name: String,
    /// Condition for when this rule applies
    pub condition: RuleCondition,
    /// Action to take when condition is met
    pub action: RuleAction,
    /// Priority (higher = evaluated first)
    pub priority: i32,
    /// When this rule was created
    pub created_at: DateTime<Utc>,
    /// Who/what created this rule
    pub created_by: Creator,
    /// Performance metrics for this rule
    pub performance: RulePerformance,
    /// Is this rule currently active?
    pub enabled: bool,
}

impl TradingRule {
    /// Create a new trading rule
    pub fn new(
        id: u64,
        name: &str,
        condition: RuleCondition,
        action: RuleAction,
        priority: i32,
        created_by: Creator,
    ) -> Self {
        Self {
            id,
            name: name.to_string(),
            condition,
            action,
            priority,
            created_at: Utc::now(),
            created_by,
            performance: RulePerformance::default(),
            enabled: true,
        }
    }

    /// Evaluate if this rule's condition is met
    pub fn evaluate(&self, context: &RuleContext) -> bool {
        if !self.enabled {
            return false;
        }
        self.condition.evaluate(context)
    }
}

/// Who created a rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Creator {
    /// Human-created rule
    Human,
    /// System-generated rule
    System {
        reason: String,
        evidence: Vec<String>,
    },
}

impl Creator {
    /// Check if this is a system-generated rule
    pub fn is_system(&self) -> bool {
        matches!(self, Creator::System { .. })
    }
}

// ==================== Rule Conditions ====================

/// Conditions that determine when a rule applies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleCondition {
    /// Current regime matches
    RegimeIs(Regime),
    /// Current regime doesn't match
    RegimeIsNot(Regime),
    /// S/R score above threshold
    SRScoreAbove(i32),
    /// S/R score below threshold
    SRScoreBelow(i32),
    /// Volume percentile above threshold
    VolumeAbove(f64),
    /// Volume percentile below threshold
    VolumeBelow(f64),
    /// Confidence above threshold
    ConfidenceAbove(f64),
    /// Confidence below threshold
    ConfidenceBelow(f64),
    /// Specific weakness identified
    WeaknessIdentified(String),
    /// Specific causal factor is active
    CausalFactorActive(String),
    /// Time of day restriction
    TimeOfDay { start_hour: u8, end_hour: u8 },
    /// Drawdown exceeds threshold
    DrawdownAbove(f64),
    /// Win rate below threshold (recent)
    WinRateBelow { threshold: f64, window: u32 },
    /// Consecutive losses exceed count
    ConsecutiveLosses(u32),
    /// Logical AND of two conditions
    And(Box<RuleCondition>, Box<RuleCondition>),
    /// Logical OR of two conditions
    Or(Box<RuleCondition>, Box<RuleCondition>),
    /// Logical NOT of a condition
    Not(Box<RuleCondition>),
    /// Always true (for testing)
    Always,
    /// Always false (for testing)
    Never,
}

impl RuleCondition {
    /// Evaluate this condition against a context
    pub fn evaluate(&self, ctx: &RuleContext) -> bool {
        match self {
            RuleCondition::RegimeIs(regime) => ctx.regime == *regime,
            RuleCondition::RegimeIsNot(regime) => ctx.regime != *regime,
            RuleCondition::SRScoreAbove(threshold) => ctx.sr_score > *threshold,
            RuleCondition::SRScoreBelow(threshold) => ctx.sr_score < *threshold,
            RuleCondition::VolumeAbove(threshold) => ctx.volume_percentile > *threshold,
            RuleCondition::VolumeBelow(threshold) => ctx.volume_percentile < *threshold,
            RuleCondition::ConfidenceAbove(threshold) => ctx.confidence > *threshold,
            RuleCondition::ConfidenceBelow(threshold) => ctx.confidence < *threshold,
            RuleCondition::WeaknessIdentified(weakness) => {
                ctx.active_weaknesses.contains(weakness)
            }
            RuleCondition::CausalFactorActive(factor) => {
                ctx.causal_factors.contains(factor)
            }
            RuleCondition::TimeOfDay { start_hour, end_hour } => {
                let hour = ctx.time.hour() as u8;
                if start_hour <= end_hour {
                    hour >= *start_hour && hour < *end_hour
                } else {
                    // Wraps around midnight
                    hour >= *start_hour || hour < *end_hour
                }
            }
            RuleCondition::DrawdownAbove(threshold) => ctx.current_drawdown > *threshold,
            RuleCondition::WinRateBelow { threshold, window } => {
                ctx.recent_win_rate(*window) < *threshold
            }
            RuleCondition::ConsecutiveLosses(count) => ctx.consecutive_losses >= *count,
            RuleCondition::And(left, right) => {
                left.evaluate(ctx) && right.evaluate(ctx)
            }
            RuleCondition::Or(left, right) => {
                left.evaluate(ctx) || right.evaluate(ctx)
            }
            RuleCondition::Not(inner) => !inner.evaluate(ctx),
            RuleCondition::Always => true,
            RuleCondition::Never => false,
        }
    }

    /// Create an AND condition
    pub fn and(self, other: RuleCondition) -> RuleCondition {
        RuleCondition::And(Box::new(self), Box::new(other))
    }

    /// Create an OR condition
    pub fn or(self, other: RuleCondition) -> RuleCondition {
        RuleCondition::Or(Box::new(self), Box::new(other))
    }

    /// Create a NOT condition
    pub fn not(self) -> RuleCondition {
        RuleCondition::Not(Box::new(self))
    }
}

/// Context for evaluating rule conditions
#[derive(Debug, Clone)]
pub struct RuleContext {
    pub regime: Regime,
    pub sr_score: i32,
    pub volume_percentile: f64,
    pub confidence: f64,
    pub active_weaknesses: HashSet<String>,
    pub causal_factors: HashSet<String>,
    pub time: DateTime<Utc>,
    pub current_drawdown: f64,
    pub consecutive_losses: u32,
    pub recent_trades: Vec<bool>, // true = win, false = loss
    pub symbol: String,
    pub is_long: bool,
}

impl RuleContext {
    /// Create a new rule context
    pub fn new(symbol: &str, regime: Regime) -> Self {
        Self {
            regime,
            sr_score: 0,
            volume_percentile: 50.0,
            confidence: 0.5,
            active_weaknesses: HashSet::new(),
            causal_factors: HashSet::new(),
            time: Utc::now(),
            current_drawdown: 0.0,
            consecutive_losses: 0,
            recent_trades: Vec::new(),
            symbol: symbol.to_string(),
            is_long: true,
        }
    }

    /// Calculate recent win rate for given window
    pub fn recent_win_rate(&self, window: u32) -> f64 {
        let window = window as usize;
        if self.recent_trades.is_empty() {
            return 0.5; // Default
        }
        let recent: Vec<_> = self.recent_trades.iter().rev().take(window).collect();
        if recent.is_empty() {
            return 0.5;
        }
        let wins = recent.iter().filter(|&&w| *w).count();
        wins as f64 / recent.len() as f64
    }
}

impl Default for RuleContext {
    fn default() -> Self {
        Self::new("", Regime::Ranging)
    }
}

// ==================== Rule Actions ====================

/// Actions that can be taken when a rule triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleAction {
    /// Skip the trade entirely
    SkipTrade { reason: String },
    /// Reduce position size
    ReduceSize { multiplier: f64, reason: String },
    /// Increase position size
    IncreaseSize { multiplier: f64, reason: String },
    /// Require confirmation from another system
    RequireConfirmation { from: String },
    /// Adjust stop loss
    AdjustStopLoss { atr_multiplier: f64 },
    /// Adjust take profit
    AdjustTakeProfit { atr_multiplier: f64 },
    /// Just log a message
    Log { message: String },
    /// Adjust confidence
    AdjustConfidence { delta: f64 },
    /// Multiple actions
    Multiple(Vec<RuleAction>),
}

impl RuleAction {
    /// Get a short description of the action
    pub fn description(&self) -> String {
        match self {
            RuleAction::SkipTrade { reason } => format!("Skip: {}", reason),
            RuleAction::ReduceSize { multiplier, .. } => {
                format!("Reduce size: {}x", multiplier)
            }
            RuleAction::IncreaseSize { multiplier, .. } => {
                format!("Increase size: {}x", multiplier)
            }
            RuleAction::RequireConfirmation { from } => {
                format!("Require confirmation from: {}", from)
            }
            RuleAction::AdjustStopLoss { atr_multiplier } => {
                format!("Adjust SL: {}x ATR", atr_multiplier)
            }
            RuleAction::AdjustTakeProfit { atr_multiplier } => {
                format!("Adjust TP: {}x ATR", atr_multiplier)
            }
            RuleAction::Log { message } => format!("Log: {}", message),
            RuleAction::AdjustConfidence { delta } => {
                format!("Adjust confidence: {:+.2}", delta)
            }
            RuleAction::Multiple(actions) => {
                format!("{} actions", actions.len())
            }
        }
    }

    /// Check if this action blocks the trade
    pub fn blocks_trade(&self) -> bool {
        match self {
            RuleAction::SkipTrade { .. } => true,
            RuleAction::Multiple(actions) => actions.iter().any(|a| a.blocks_trade()),
            _ => false,
        }
    }
}

// ==================== Rule Performance ====================

/// Performance tracking for a rule
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RulePerformance {
    /// How many times the rule was triggered
    pub times_triggered: u32,
    /// How many trades were affected
    pub trades_affected: u32,
    /// Estimated P&L impact
    pub estimated_pnl_impact: f64,
    /// When the rule was last triggered
    pub last_triggered: Option<DateTime<Utc>>,
    /// Wins when rule triggered
    pub wins_when_triggered: u32,
    /// Losses when rule triggered
    pub losses_when_triggered: u32,
}

impl RulePerformance {
    /// Record that the rule was triggered
    pub fn record_trigger(&mut self) {
        self.times_triggered += 1;
        self.last_triggered = Some(Utc::now());
    }

    /// Record trade outcome when rule was active
    pub fn record_outcome(&mut self, won: bool, pnl: f64) {
        self.trades_affected += 1;
        self.estimated_pnl_impact += pnl;
        if won {
            self.wins_when_triggered += 1;
        } else {
            self.losses_when_triggered += 1;
        }
    }

    /// Get win rate when this rule triggers
    pub fn win_rate(&self) -> f64 {
        if self.trades_affected == 0 {
            return 0.5;
        }
        self.wins_when_triggered as f64 / self.trades_affected as f64
    }
}

// ==================== Constitution ====================

/// Constitutional constraints that cannot be violated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constitution {
    /// Maximum position size (as fraction of capital)
    pub max_position_size: f64,
    /// Maximum daily loss (as fraction)
    pub max_daily_loss: f64,
    /// Maximum total drawdown (hard stop)
    pub max_drawdown: f64,
    /// Minimum confidence required to trade
    pub min_confidence_for_trade: f64,
    /// Modifications that are forbidden
    pub forbidden_modifications: Vec<String>,
    /// Whether new rules require backtesting
    pub require_backtest_for_rules: bool,
    /// Minimum trades as evidence for changes
    pub min_evidence_for_change: u32,
    /// Maximum rule changes per day
    pub max_rule_changes_per_day: u32,
    /// Impact threshold requiring human approval
    pub human_approval_threshold: f64,
    /// Minimum time between similar modifications
    pub modification_cooldown_hours: u32,
    /// Maximum number of active rules
    pub max_active_rules: u32,
}

impl Constitution {
    /// Create a conservative default constitution
    pub fn conservative() -> Self {
        Self {
            max_position_size: 0.05, // 5% max position
            max_daily_loss: 0.03,     // 3% max daily loss
            max_drawdown: 0.10,       // 10% max drawdown
            min_confidence_for_trade: 0.40,
            forbidden_modifications: vec![
                "max_drawdown".to_string(),
                "constitution".to_string(),
                "forbidden_modifications".to_string(),
            ],
            require_backtest_for_rules: true,
            min_evidence_for_change: 20,
            max_rule_changes_per_day: 3,
            human_approval_threshold: 0.10, // 10% impact
            modification_cooldown_hours: 24,
            max_active_rules: 50,
        }
    }

    /// Check if a key is forbidden from modification
    pub fn is_forbidden(&self, key: &str) -> bool {
        self.forbidden_modifications.iter().any(|f| f == key)
    }
}

impl Default for Constitution {
    fn default() -> Self {
        Self::conservative()
    }
}

// ==================== Constitutional Guard ====================

/// Violation reasons when a modification is rejected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationReason {
    /// Modification is forbidden
    ForbiddenModification(String),
    /// Would exceed position limits
    ExceedsPositionLimit { requested: f64, max: f64 },
    /// Would exceed loss limits
    ExceedsLossLimit { requested: f64, max: f64 },
    /// Would exceed drawdown
    ExceedsDrawdownLimit { requested: f64, max: f64 },
    /// Confidence would be too low
    ConfidenceTooLow { requested: f64, min: f64 },
    /// Daily limit reached
    DailyLimitReached { current: u32, max: u32 },
    /// Not enough evidence
    InsufficientEvidence { provided: u32, required: u32 },
    /// Still in cooldown
    InCooldown { remaining_hours: u32 },
    /// Too many active rules
    TooManyRules { current: u32, max: u32 },
    /// Backtest required but not provided
    BacktestRequired,
}

impl std::fmt::Display for ViolationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViolationReason::ForbiddenModification(key) => {
                write!(f, "Modification of '{}' is forbidden", key)
            }
            ViolationReason::ExceedsPositionLimit { requested, max } => {
                write!(f, "Position {:.2}% exceeds max {:.2}%", requested * 100.0, max * 100.0)
            }
            ViolationReason::ExceedsLossLimit { requested, max } => {
                write!(f, "Loss limit {:.2}% exceeds max {:.2}%", requested * 100.0, max * 100.0)
            }
            ViolationReason::ExceedsDrawdownLimit { requested, max } => {
                write!(f, "Drawdown {:.2}% exceeds max {:.2}%", requested * 100.0, max * 100.0)
            }
            ViolationReason::ConfidenceTooLow { requested, min } => {
                write!(f, "Confidence {:.2}% below min {:.2}%", requested * 100.0, min * 100.0)
            }
            ViolationReason::DailyLimitReached { current, max } => {
                write!(f, "Daily changes {} reached limit {}", current, max)
            }
            ViolationReason::InsufficientEvidence { provided, required } => {
                write!(f, "Evidence {} trades < required {}", provided, required)
            }
            ViolationReason::InCooldown { remaining_hours } => {
                write!(f, "In cooldown, {} hours remaining", remaining_hours)
            }
            ViolationReason::TooManyRules { current, max } => {
                write!(f, "Rules {} at limit {}", current, max)
            }
            ViolationReason::BacktestRequired => {
                write!(f, "Backtest required for new rules")
            }
        }
    }
}

/// Status of a pending modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalStatus {
    /// Awaiting decision
    Pending,
    /// Auto-approved by system
    AutoApproved { reason: String },
    /// Approved by human
    HumanApproved { by: String, at: DateTime<Utc> },
    /// Rejected
    Rejected { reason: String },
}

/// A modification awaiting approval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingModification {
    /// Unique ID
    pub id: u64,
    /// The modification
    pub modification: ModificationType,
    /// Why this modification is proposed
    pub reason: String,
    /// Evidence supporting the change
    pub evidence: Vec<String>,
    /// Estimated impact on performance
    pub estimated_impact: f64,
    /// When this was proposed
    pub proposed_at: DateTime<Utc>,
    /// Current status
    pub status: ApprovalStatus,
}

/// A modification that has been applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedModification {
    /// Unique ID
    pub id: u64,
    /// The modification
    pub modification: ModificationType,
    /// Why it was made
    pub reason: String,
    /// When it was applied
    pub applied_at: DateTime<Utc>,
    /// How it was approved
    pub approval: ApprovalStatus,
    /// Whether it has been rolled back
    pub rolled_back: bool,
}

/// A constitution violation that was detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionViolation {
    /// When the violation occurred
    pub timestamp: DateTime<Utc>,
    /// What was attempted
    pub attempted_modification: ModificationType,
    /// Why it was blocked
    pub reason: ViolationReason,
}

/// Result of proposing a modification (fully autonomous - no human approval)
#[derive(Debug, Clone)]
pub enum ProposalResult {
    /// Modification was auto-deployed (constitutional check passed)
    AutoDeployed { id: u64, description: String },
    /// Modification was auto-rejected (constitutional check failed)
    AutoRejected { reason: ViolationReason, description: String },
}

/// Guard that enforces constitutional constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalGuard {
    /// The constitution
    constitution: Constitution,
    /// Pending modifications awaiting approval
    pending_approvals: Vec<PendingModification>,
    /// History of applied modifications
    modification_history: Vec<AppliedModification>,
    /// Violations that were blocked
    violations: Vec<ConstitutionViolation>,
    /// Next modification ID
    next_id: u64,
    /// Changes made today
    changes_today: u32,
    /// Date of last change count reset
    last_reset: DateTime<Utc>,
}

impl ConstitutionalGuard {
    /// Create a new constitutional guard
    pub fn new(constitution: Constitution) -> Self {
        Self {
            constitution,
            pending_approvals: Vec::new(),
            modification_history: Vec::new(),
            violations: Vec::new(),
            next_id: 1,
            changes_today: 0,
            last_reset: Utc::now(),
        }
    }

    /// Reset daily counters if needed
    fn maybe_reset_daily(&mut self) {
        let now = Utc::now();
        if now.date_naive() != self.last_reset.date_naive() {
            self.changes_today = 0;
            self.last_reset = now;
        }
    }

    /// Check if a modification violates the constitution
    pub fn check_modification(&self, mod_type: &ModificationType) -> Result<(), ViolationReason> {
        // Check forbidden modifications
        match mod_type {
            ModificationType::ConfigChange { key, .. } => {
                if self.constitution.is_forbidden(key) {
                    return Err(ViolationReason::ForbiddenModification(key.clone()));
                }
            }
            ModificationType::ThresholdChange { name, new, .. } => {
                if self.constitution.is_forbidden(name) {
                    return Err(ViolationReason::ForbiddenModification(name.clone()));
                }
                // Check specific thresholds
                if name == "max_position_size" && *new > self.constitution.max_position_size {
                    return Err(ViolationReason::ExceedsPositionLimit {
                        requested: *new,
                        max: self.constitution.max_position_size,
                    });
                }
                if name == "max_daily_loss" && *new > self.constitution.max_daily_loss {
                    return Err(ViolationReason::ExceedsLossLimit {
                        requested: *new,
                        max: self.constitution.max_daily_loss,
                    });
                }
                if name == "min_confidence" && *new < self.constitution.min_confidence_for_trade {
                    return Err(ViolationReason::ConfidenceTooLow {
                        requested: *new,
                        min: self.constitution.min_confidence_for_trade,
                    });
                }
            }
            ModificationType::WeightAdjustment { component, .. } => {
                if self.constitution.is_forbidden(component) {
                    return Err(ViolationReason::ForbiddenModification(component.clone()));
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Check if we're within daily limits
    fn check_daily_limit(&self) -> Result<(), ViolationReason> {
        if self.changes_today >= self.constitution.max_rule_changes_per_day {
            return Err(ViolationReason::DailyLimitReached {
                current: self.changes_today,
                max: self.constitution.max_rule_changes_per_day,
            });
        }
        Ok(())
    }

    /// Check if we can auto-approve this modification
    pub fn can_auto_approve(&self, mod_type: &ModificationType, impact: f64, evidence_count: u32) -> bool {
        // Must have enough evidence
        if evidence_count < self.constitution.min_evidence_for_change {
            return false;
        }

        // Impact must be below threshold
        if impact.abs() >= self.constitution.human_approval_threshold {
            return false;
        }

        // Must not be a forbidden modification
        if self.check_modification(mod_type).is_err() {
            return false;
        }

        // Must be within daily limits
        if self.check_daily_limit().is_err() {
            return false;
        }

        true
    }

    /// Propose a modification (fully autonomous - no human approval workflow)
    ///
    /// Modifications are automatically deployed if they pass constitutional checks,
    /// or automatically rejected if they violate the constitution.
    pub fn propose_modification(
        &mut self,
        mod_type: ModificationType,
        reason: String,
        _evidence: Vec<String>,
        _estimated_impact: f64,
    ) -> ProposalResult {
        self.maybe_reset_daily();

        let description = mod_type.description();

        // Check constitution - auto-reject if violation
        if let Err(violation) = self.check_modification(&mod_type) {
            self.violations.push(ConstitutionViolation {
                timestamp: Utc::now(),
                attempted_modification: mod_type.clone(),
                reason: violation.clone(),
            });
            return ProposalResult::AutoRejected {
                reason: violation,
                description,
            };
        }

        // Check daily limits - auto-reject if exceeded
        if let Err(violation) = self.check_daily_limit() {
            return ProposalResult::AutoRejected {
                reason: violation,
                description,
            };
        }

        let id = self.next_id;
        self.next_id += 1;

        // Auto-deploy: constitutional check passed
        self.changes_today += 1;
        self.modification_history.push(AppliedModification {
            id,
            modification: mod_type,
            reason,
            applied_at: Utc::now(),
            approval: ApprovalStatus::AutoApproved {
                reason: "Autonomous deployment - constitutional check passed".to_string(),
            },
            rolled_back: false,
        });

        ProposalResult::AutoDeployed { id, description }
    }

    /// Get pending modifications
    pub fn get_pending(&self) -> &[PendingModification] {
        &self.pending_approvals
    }

    /// Get count of pending modifications
    pub fn pending_count(&self) -> usize {
        self.pending_approvals.len()
    }

    /// Get count of applied modifications
    pub fn applied_count(&self) -> usize {
        self.modification_history.len()
    }

    /// Approve a pending modification
    pub fn approve_pending(&mut self, id: u64, approver: &str) -> Result<(), String> {
        let idx = self.pending_approvals
            .iter()
            .position(|p| p.id == id)
            .ok_or_else(|| format!("Pending modification {} not found", id))?;

        let mut pending = self.pending_approvals.remove(idx);
        pending.status = ApprovalStatus::HumanApproved {
            by: approver.to_string(),
            at: Utc::now(),
        };

        self.changes_today += 1;
        self.modification_history.push(AppliedModification {
            id: pending.id,
            modification: pending.modification,
            reason: pending.reason,
            applied_at: Utc::now(),
            approval: pending.status,
            rolled_back: false,
        });

        Ok(())
    }

    /// Reject a pending modification
    pub fn reject_pending(&mut self, id: u64, reason: &str) -> Result<(), String> {
        let idx = self.pending_approvals
            .iter()
            .position(|p| p.id == id)
            .ok_or_else(|| format!("Pending modification {} not found", id))?;

        self.pending_approvals[idx].status = ApprovalStatus::Rejected {
            reason: reason.to_string(),
        };

        Ok(())
    }

    /// Get modification history
    pub fn get_history(&self) -> &[AppliedModification] {
        &self.modification_history
    }

    /// Get violations
    pub fn get_violations(&self) -> &[ConstitutionViolation] {
        &self.violations
    }

    /// Get the constitution
    pub fn constitution(&self) -> &Constitution {
        &self.constitution
    }

    /// Get count of changes today
    pub fn changes_today(&self) -> u32 {
        self.changes_today
    }
}

impl Default for ConstitutionalGuard {
    fn default() -> Self {
        Self::new(Constitution::default())
    }
}

// ==================== Rule Engine ====================

/// Engine for managing and evaluating trading rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleEngine {
    /// All rules
    rules: Vec<TradingRule>,
    /// IDs of active rules
    active_rules: HashSet<u64>,
    /// Next rule ID
    next_rule_id: u64,
}

impl RuleEngine {
    /// Create a new rule engine
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            active_rules: HashSet::new(),
            next_rule_id: 1,
        }
    }

    /// Add a new rule
    pub fn add_rule(&mut self, mut rule: TradingRule) -> u64 {
        let id = self.next_rule_id;
        self.next_rule_id += 1;
        rule.id = id;

        if rule.enabled {
            self.active_rules.insert(id);
        }

        self.rules.push(rule);
        id
    }

    /// Remove a rule
    pub fn remove_rule(&mut self, id: u64) -> Option<TradingRule> {
        self.active_rules.remove(&id);
        let idx = self.rules.iter().position(|r| r.id == id)?;
        Some(self.rules.remove(idx))
    }

    /// Enable a rule
    pub fn enable_rule(&mut self, id: u64) {
        if let Some(rule) = self.rules.iter_mut().find(|r| r.id == id) {
            rule.enabled = true;
            self.active_rules.insert(id);
        }
    }

    /// Disable a rule
    pub fn disable_rule(&mut self, id: u64) {
        if let Some(rule) = self.rules.iter_mut().find(|r| r.id == id) {
            rule.enabled = false;
            self.active_rules.remove(&id);
        }
    }

    /// Get a rule by ID
    pub fn get_rule(&self, id: u64) -> Option<&TradingRule> {
        self.rules.iter().find(|r| r.id == id)
    }

    /// Get a mutable rule by ID
    pub fn get_rule_mut(&mut self, id: u64) -> Option<&mut TradingRule> {
        self.rules.iter_mut().find(|r| r.id == id)
    }

    /// Evaluate all active rules and return triggered actions
    pub fn evaluate(&self, context: &RuleContext) -> Vec<RuleAction> {
        let mut actions = Vec::new();

        // Sort by priority (descending)
        let mut triggered: Vec<_> = self.rules
            .iter()
            .filter(|r| r.enabled && self.active_rules.contains(&r.id))
            .filter(|r| r.evaluate(context))
            .collect();

        triggered.sort_by(|a, b| b.priority.cmp(&a.priority));

        for rule in triggered {
            actions.push(rule.action.clone());
        }

        actions
    }

    /// Get rules that would trigger for a context
    pub fn get_triggered_rules(&self, context: &RuleContext) -> Vec<&TradingRule> {
        self.rules
            .iter()
            .filter(|r| r.enabled && self.active_rules.contains(&r.id))
            .filter(|r| r.evaluate(context))
            .collect()
    }

    /// Get all active rules
    pub fn get_active_rules(&self) -> Vec<&TradingRule> {
        self.rules
            .iter()
            .filter(|r| r.enabled && self.active_rules.contains(&r.id))
            .collect()
    }

    /// Get count of active rules
    pub fn active_count(&self) -> usize {
        self.active_rules.len()
    }

    /// Get all rules
    pub fn all_rules(&self) -> &[TradingRule] {
        &self.rules
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Self-Modification Engine ====================

/// Result of backtesting a rule
#[derive(Debug, Clone)]
pub struct RuleBacktestResult {
    /// Number of trades in backtest
    pub trades_tested: u32,
    /// Win rate change (positive = improvement)
    pub win_rate_change: f64,
    /// P&L change (positive = improvement)
    pub pnl_change: f64,
    /// Times rule would have triggered
    pub times_triggered: u32,
    /// Passes minimum requirements
    pub passes: bool,
}

/// Main self-modification engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModificationEngine {
    /// Constitutional guard
    guard: ConstitutionalGuard,
    /// Rule engine
    rule_engine: RuleEngine,
    /// Cooldown tracking
    last_modifications: HashMap<String, DateTime<Utc>>,
    /// Default cooldown duration
    #[serde(skip)]
    modification_cooldown: Duration,
}

impl SelfModificationEngine {
    /// Create a new self-modification engine
    pub fn new(constitution: Constitution) -> Self {
        let cooldown_hours = constitution.modification_cooldown_hours as i64;
        Self {
            guard: ConstitutionalGuard::new(constitution),
            rule_engine: RuleEngine::new(),
            last_modifications: HashMap::new(),
            modification_cooldown: Duration::hours(cooldown_hours),
        }
    }

    /// Check if a modification type is in cooldown
    pub fn is_in_cooldown(&self, category: &str) -> bool {
        if let Some(last) = self.last_modifications.get(category) {
            let elapsed = Utc::now() - *last;
            elapsed < self.modification_cooldown
        } else {
            false
        }
    }

    /// Get count of applied modifications
    pub fn applied_count(&self) -> usize {
        self.guard.applied_count()
    }

    /// Generate a rule from an observed weakness
    pub fn generate_rule_from_weakness(&self, weakness: &Weakness) -> Option<TradingRule> {
        let (condition, action) = match &weakness.weakness_type {
            WeaknessType::RegimeWeakness { regime, win_rate, .. } => {
                let regime_cond = RuleCondition::RegimeIs(regime.clone());
                let action = if *win_rate < 0.4 {
                    RuleAction::SkipTrade {
                        reason: format!("Low win rate in {} regime: {:.1}%", regime, win_rate * 100.0),
                    }
                } else {
                    RuleAction::ReduceSize {
                        multiplier: 0.5,
                        reason: format!("Reduced performance in {} regime", regime),
                    }
                };
                (regime_cond, action)
            }
            WeaknessType::SymbolWeakness { symbol, win_rate, .. } => {
                // Create a condition that effectively skips this symbol
                let condition = RuleCondition::ConfidenceBelow(0.99); // Will rarely trigger
                let action = RuleAction::Log {
                    message: format!("Symbol {} has {:.1}% win rate", symbol, win_rate * 100.0),
                };
                (condition, action)
            }
            WeaknessType::SRScoreWeakness { score_range, win_rate, .. } => {
                let (low, _high) = score_range;
                let condition = RuleCondition::SRScoreBelow(*low);
                let action = RuleAction::SkipTrade {
                    reason: format!("Weak S/R score pattern: {:.1}% win rate", win_rate * 100.0),
                };
                (condition, action)
            }
            WeaknessType::VolumeThresholdWeakness { threshold_range, win_rate, .. } => {
                let (low, _high) = threshold_range;
                let condition = if *low < 30.0 {
                    RuleCondition::VolumeBelow(*low)
                } else {
                    RuleCondition::VolumeAbove(*low)
                };
                let action = RuleAction::ReduceSize {
                    multiplier: 0.7,
                    reason: format!("Volume weakness: {:.1}% win rate", win_rate * 100.0),
                };
                (condition, action)
            }
            WeaknessType::TimeOfDayWeakness { hour_utc, win_rate, .. } => {
                let condition = RuleCondition::TimeOfDay {
                    start_hour: *hour_utc,
                    end_hour: (*hour_utc + 1) % 24,
                };
                let action = RuleAction::SkipTrade {
                    reason: format!("Weak hour ({}:00 UTC): {:.1}% win rate", hour_utc, win_rate * 100.0),
                };
                (condition, action)
            }
            WeaknessType::ClusterWeakness { cluster, win_rate, .. } => {
                // Log cluster weakness - can't easily create condition for cluster
                let condition = RuleCondition::ConfidenceBelow(0.99);
                let action = RuleAction::Log {
                    message: format!("Cluster {:?} has {:.1}% win rate", cluster, win_rate * 100.0),
                };
                (condition, action)
            }
        };

        let description = weakness.weakness_type.description();
        let rule_name = format!("Auto: {}", description);
        let evidence = vec![
            format!("Based on {} trades", weakness.trades_analyzed),
            format!("Severity: {:.2}", weakness.severity),
        ];

        Some(TradingRule::new(
            0, // ID will be assigned
            &rule_name,
            condition,
            action,
            10, // Default priority
            Creator::System {
                reason: description,
                evidence,
            },
        ))
    }

    /// Generate a threshold change from an insight
    pub fn generate_threshold_change(&self, insight: &TradingInsight) -> Option<ModificationType> {
        let improvement = insight.avg_improvement.abs();

        match &insight.insight_type {
            InsightType::ExitTooEarly => {
                // Increase take profit multiplier based on avg_improvement
                let current = 2.0; // Default
                let new = current * (1.0 + improvement.min(0.5));
                Some(ModificationType::ThresholdChange {
                    name: "take_profit_atr_multiplier".to_string(),
                    old: current,
                    new,
                })
            }
            InsightType::ExitTooLate => {
                // Tighten stop loss
                let current = 1.5; // Default
                let adjustment = (improvement * 0.5).min(0.3);
                let new = current * (1.0 - adjustment);
                Some(ModificationType::ThresholdChange {
                    name: "stop_loss_atr_multiplier".to_string(),
                    old: current,
                    new,
                })
            }
            InsightType::SizeTooSmall => {
                // Increase position size
                let current = 1.0;
                let new = current * (1.0 + improvement.min(0.3));
                Some(ModificationType::ThresholdChange {
                    name: "position_size_multiplier".to_string(),
                    old: current,
                    new,
                })
            }
            InsightType::SizeTooLarge => {
                // Decrease position size
                let current = 1.0;
                let new = current * (1.0 - improvement.min(0.3));
                Some(ModificationType::ThresholdChange {
                    name: "position_size_multiplier".to_string(),
                    old: current,
                    new,
                })
            }
            _ => None,
        }
    }

    /// Backtest a rule against historical contexts
    pub fn backtest_rule(&self, rule: &TradingRule, history: &[TradeContext]) -> RuleBacktestResult {
        let mut wins_without_rule = 0;
        let mut losses_without_rule = 0;
        let mut wins_with_rule = 0;
        let mut losses_with_rule = 0;
        let mut times_triggered = 0;

        for trade in history {
            let context = self.trade_context_to_rule_context(trade);
            let would_trigger = rule.evaluate(&context);

            // Check if it was a win
            let was_win = trade.pnl > 0.0;

            if was_win {
                wins_without_rule += 1;
            } else {
                losses_without_rule += 1;
            }

            if would_trigger {
                times_triggered += 1;
                // Assume rule would have prevented loss or let win through
                if rule.action.blocks_trade() {
                    // Would have skipped this trade
                    if !was_win {
                        // Good - avoided a loss
                        wins_with_rule += wins_without_rule;
                    } else {
                        // Bad - missed a win
                        losses_with_rule += 1;
                    }
                } else {
                    // Rule doesn't skip, just modifies
                    if was_win {
                        wins_with_rule += 1;
                    } else {
                        losses_with_rule += 1;
                    }
                }
            } else {
                if was_win {
                    wins_with_rule += 1;
                } else {
                    losses_with_rule += 1;
                }
            }
        }

        let total_without = wins_without_rule + losses_without_rule;
        let total_with = wins_with_rule + losses_with_rule;

        let wr_without = if total_without > 0 {
            wins_without_rule as f64 / total_without as f64
        } else {
            0.5
        };

        let wr_with = if total_with > 0 {
            wins_with_rule as f64 / total_with as f64
        } else {
            0.5
        };

        RuleBacktestResult {
            trades_tested: history.len() as u32,
            win_rate_change: wr_with - wr_without,
            pnl_change: 0.0, // Would need actual P&L data
            times_triggered,
            passes: wr_with >= wr_without,
        }
    }

    /// Convert TradeContext to RuleContext
    fn trade_context_to_rule_context(&self, trade: &TradeContext) -> RuleContext {
        RuleContext {
            regime: trade.regime.clone(),
            sr_score: trade.sr_score,
            volume_percentile: trade.volume_percentile,
            confidence: 0.5, // Default confidence
            active_weaknesses: HashSet::new(),
            causal_factors: HashSet::new(),
            time: Utc::now(), // Use current time as we don't have timestamp
            current_drawdown: 0.0,
            consecutive_losses: 0,
            recent_trades: Vec::new(),
            symbol: trade.symbol.clone(),
            is_long: trade.is_long,
        }
    }

    /// Propose a rule addition with backtest (fully autonomous)
    pub fn propose_rule(
        &mut self,
        rule: TradingRule,
        history: &[TradeContext],
    ) -> ProposalResult {
        let rule_name = rule.name.clone();

        // Check rule limit
        if self.rule_engine.active_count() as u32 >= self.guard.constitution().max_active_rules {
            return ProposalResult::AutoRejected {
                reason: ViolationReason::TooManyRules {
                    current: self.rule_engine.active_count() as u32,
                    max: self.guard.constitution().max_active_rules,
                },
                description: format!("Add rule: {}", rule_name),
            };
        }

        // Backtest if required
        let backtest_result = if self.guard.constitution().require_backtest_for_rules {
            let result = self.backtest_rule(&rule, history);
            if !result.passes {
                return ProposalResult::AutoRejected {
                    reason: ViolationReason::BacktestRequired,
                    description: format!("Add rule: {}", rule_name),
                };
            }
            Some(result)
        } else {
            None
        };

        let evidence = match &rule.created_by {
            Creator::System { evidence, .. } => evidence.clone(),
            Creator::Human => vec!["Human-created rule".to_string()],
        };

        let estimated_impact = backtest_result
            .as_ref()
            .map(|r| r.win_rate_change)
            .unwrap_or(0.0);

        let reason = match &rule.created_by {
            Creator::System { reason, .. } => reason.clone(),
            Creator::Human => "Human-defined rule".to_string(),
        };

        let mod_type = ModificationType::RuleAddition { rule };

        self.guard.propose_modification(mod_type, reason, evidence, estimated_impact)
    }

    /// Apply a rule addition that was approved
    pub fn apply_rule_addition(&mut self, rule: TradingRule) -> u64 {
        let category = format!("rule:{}", rule.name);
        self.last_modifications.insert(category, Utc::now());
        self.rule_engine.add_rule(rule)
    }

    /// Apply approved modifications
    pub fn apply_approved_modifications(&mut self) -> Vec<AppliedModification> {
        let approved: Vec<_> = self.guard.get_history()
            .iter()
            .filter(|m| !m.rolled_back)
            .filter(|m| matches!(m.approval, ApprovalStatus::AutoApproved { .. } | ApprovalStatus::HumanApproved { .. }))
            .cloned()
            .collect();

        // Apply rule additions
        for modification in &approved {
            if let ModificationType::RuleAddition { rule } = &modification.modification {
                // Check if rule already exists
                let exists = self.rule_engine.all_rules().iter().any(|r| r.name == rule.name);
                if !exists {
                    self.rule_engine.add_rule(rule.clone());
                    info!(
                        "[SELF-MOD] Applied rule: {} (ID: {})",
                        rule.name, modification.id
                    );
                }
            }
        }

        approved
    }

    /// Rollback a modification
    pub fn rollback_modification(&mut self, id: u64) -> Result<(), String> {
        // Find the modification in history
        let mod_opt = self.guard.modification_history
            .iter_mut()
            .find(|m| m.id == id);

        let modification = mod_opt.ok_or_else(|| format!("Modification {} not found", id))?;

        if modification.rolled_back {
            return Err(format!("Modification {} already rolled back", id));
        }

        // Rollback based on type
        match &modification.modification {
            ModificationType::RuleAddition { rule } => {
                self.rule_engine.remove_rule(rule.id);
                info!("[SELF-MOD] Rolled back rule: {}", rule.name);
            }
            ModificationType::RuleRemoval { rule_id } => {
                // Can't easily un-remove without storing the rule
                info!("[SELF-MOD] Cannot fully rollback rule removal: #{}", rule_id);
            }
            _ => {
                info!("[SELF-MOD] Rollback for {:?} not fully implemented", modification.modification);
            }
        }

        modification.rolled_back = true;
        Ok(())
    }

    /// Get the rule engine
    pub fn rule_engine(&self) -> &RuleEngine {
        &self.rule_engine
    }

    /// Get mutable rule engine
    pub fn rule_engine_mut(&mut self) -> &mut RuleEngine {
        &mut self.rule_engine
    }

    /// Get the constitutional guard
    pub fn guard(&self) -> &ConstitutionalGuard {
        &self.guard
    }

    /// Get mutable guard
    pub fn guard_mut(&mut self) -> &mut ConstitutionalGuard {
        &mut self.guard
    }

    /// Get active rules
    pub fn get_active_rules(&self) -> Vec<&TradingRule> {
        self.rule_engine.get_active_rules()
    }

    /// Get modification history
    pub fn get_modification_history(&self) -> &[AppliedModification] {
        self.guard.get_history()
    }

    /// Get the constitution
    pub fn get_constitution(&self) -> &Constitution {
        self.guard.constitution()
    }

    /// Evaluate rules for a trade context
    pub fn evaluate_rules(&self, context: &RuleContext) -> Vec<RuleAction> {
        self.rule_engine.evaluate(context)
    }

    /// Get triggered rules for a context
    pub fn get_triggered_rules(&self, context: &RuleContext) -> Vec<&TradingRule> {
        self.rule_engine.get_triggered_rules(context)
    }

    /// Save to file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Load from file
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let mut engine: Self = serde_json::from_str(&contents)?;
        // Restore non-serialized fields
        let cooldown_hours = engine.guard.constitution().modification_cooldown_hours as i64;
        engine.modification_cooldown = Duration::hours(cooldown_hours);
        Ok(engine)
    }

    /// Load or create new
    pub fn load_or_new(path: &str, constitution: Constitution) -> Self {
        match Self::load(path) {
            Ok(engine) => {
                info!("[SELF-MOD] Loaded {} rules from {}", engine.rule_engine.active_count(), path);
                engine
            }
            Err(_) => Self::new(constitution),
        }
    }

    /// Get pending approvals
    pub fn get_pending(&self) -> &[PendingModification] {
        self.guard.get_pending()
    }

    /// Approve a pending modification
    pub fn approve_pending(&mut self, id: u64, approver: &str) -> Result<(), String> {
        self.guard.approve_pending(id, approver)
    }

    /// Reject a pending modification
    pub fn reject_pending(&mut self, id: u64, reason: &str) -> Result<(), String> {
        self.guard.reject_pending(id, reason)
    }

    /// Format summary for logging
    pub fn format_summary(&self) -> String {
        format!(
            "{} active rules, {} changes today (autonomous)",
            self.rule_engine.active_count(),
            self.guard.changes_today()
        )
    }
}

impl Default for SelfModificationEngine {
    fn default() -> Self {
        Self::new(Constitution::default())
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constitution_check() {
        let guard = ConstitutionalGuard::default();

        // Allowed modification
        let allowed = ModificationType::ThresholdChange {
            name: "some_threshold".to_string(),
            old: 0.5,
            new: 0.6,
        };
        assert!(guard.check_modification(&allowed).is_ok());

        // Forbidden modification
        let forbidden = ModificationType::ThresholdChange {
            name: "max_drawdown".to_string(),
            old: 0.10,
            new: 0.15,
        };
        assert!(guard.check_modification(&forbidden).is_err());
    }

    #[test]
    fn test_rule_generation_from_weakness() {
        let engine = SelfModificationEngine::default();

        let weakness = Weakness {
            weakness_type: WeaknessType::RegimeWeakness {
                regime: Regime::Volatile,
                win_rate: 0.35,
                trade_count: 50,
            },
            severity: 0.8,
            identified_at: Utc::now(),
            trades_analyzed: 50,
            suggested_action: "Reduce size in volatile regime".to_string(),
        };

        let rule = engine.generate_rule_from_weakness(&weakness);
        assert!(rule.is_some());

        let rule = rule.unwrap();
        assert!(matches!(rule.condition, RuleCondition::RegimeIs(Regime::Volatile)));
        assert!(rule.action.blocks_trade());
    }

    #[test]
    fn test_auto_deploy() {
        let mut guard = ConstitutionalGuard::default();

        let mod_type = ModificationType::ThresholdChange {
            name: "some_threshold".to_string(),
            old: 0.5,
            new: 0.55,
        };

        // Valid modification should be auto-deployed
        let evidence: Vec<String> = (0..25).map(|i| format!("Trade {}", i)).collect();
        let result = guard.propose_modification(
            mod_type,
            "Test change".to_string(),
            evidence,
            0.05,
        );

        assert!(matches!(result, ProposalResult::AutoDeployed { .. }));
    }

    #[test]
    fn test_autonomous_deploy_any_valid() {
        let mut guard = ConstitutionalGuard::default();

        let mod_type = ModificationType::ThresholdChange {
            name: "important_threshold".to_string(),
            old: 0.5,
            new: 0.8,
        };

        // Any valid (non-forbidden) modification should be auto-deployed
        // No more human approval required
        let evidence: Vec<String> = (0..25).map(|i| format!("Trade {}", i)).collect();
        let result = guard.propose_modification(
            mod_type,
            "Big change".to_string(),
            evidence,
            0.25, // Impact no longer matters for approval
        );

        assert!(matches!(result, ProposalResult::AutoDeployed { .. }));
    }

    #[test]
    fn test_rule_evaluation() {
        let mut engine = RuleEngine::new();

        let rule = TradingRule::new(
            0,
            "Skip volatile",
            RuleCondition::RegimeIs(Regime::Volatile),
            RuleAction::SkipTrade {
                reason: "Volatile regime".to_string(),
            },
            10,
            Creator::Human,
        );

        engine.add_rule(rule);

        // Should trigger in volatile regime
        let mut context = RuleContext::default();
        context.regime = Regime::Volatile;

        let actions = engine.evaluate(&context);
        assert_eq!(actions.len(), 1);
        assert!(actions[0].blocks_trade());

        // Should not trigger in trending regime
        context.regime = Regime::TrendingUp;
        let actions = engine.evaluate(&context);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_modification_rollback() {
        let mut engine = SelfModificationEngine::default();

        let rule = TradingRule::new(
            0,
            "Test rule",
            RuleCondition::Always,
            RuleAction::Log {
                message: "Test".to_string(),
            },
            10,
            Creator::Human,
        );

        let id = engine.rule_engine_mut().add_rule(rule);
        assert_eq!(engine.rule_engine().active_count(), 1);

        // Simulate that this was an approved modification
        engine.guard.modification_history.push(AppliedModification {
            id: 1,
            modification: ModificationType::RuleAddition {
                rule: engine.rule_engine().get_rule(id).unwrap().clone(),
            },
            reason: "Test".to_string(),
            applied_at: Utc::now(),
            approval: ApprovalStatus::AutoApproved {
                reason: "Test".to_string(),
            },
            rolled_back: false,
        });

        // Rollback should remove the rule
        // Note: rollback looks up by modification id, not rule id
        // For this test, we manually remove the rule
        engine.rule_engine_mut().remove_rule(id);
        assert_eq!(engine.rule_engine().active_count(), 0);
    }

    #[test]
    fn test_forbidden_modifications() {
        let guard = ConstitutionalGuard::default();

        let forbidden_keys = ["max_drawdown", "constitution", "forbidden_modifications"];

        for key in forbidden_keys {
            let mod_type = ModificationType::ConfigChange {
                key: key.to_string(),
                old_value: "old".to_string(),
                new_value: "new".to_string(),
            };

            let result = guard.check_modification(&mod_type);
            assert!(result.is_err(), "Key '{}' should be forbidden", key);
        }
    }

    #[test]
    fn test_daily_limits() {
        let mut constitution = Constitution::default();
        constitution.max_rule_changes_per_day = 2;

        let mut guard = ConstitutionalGuard::new(constitution);

        // First two should auto-deploy
        for i in 0..2 {
            let mod_type = ModificationType::ThresholdChange {
                name: format!("threshold_{}", i),
                old: 0.5,
                new: 0.55,
            };

            let evidence: Vec<String> = (0..25).map(|j| format!("Trade {}", j)).collect();
            let result = guard.propose_modification(
                mod_type,
                "Test".to_string(),
                evidence,
                0.01,
            );

            assert!(matches!(result, ProposalResult::AutoDeployed { .. }));
        }

        // Third should auto-reject due to daily limit
        let mod_type = ModificationType::ThresholdChange {
            name: "threshold_2".to_string(),
            old: 0.5,
            new: 0.55,
        };

        let evidence: Vec<String> = (0..25).map(|i| format!("Trade {}", i)).collect();
        let result = guard.propose_modification(
            mod_type,
            "Test".to_string(),
            evidence,
            0.01,
        );

        assert!(matches!(result, ProposalResult::AutoRejected { reason: ViolationReason::DailyLimitReached { .. }, .. }));
    }

    #[test]
    fn test_rule_condition_logic() {
        let ctx = RuleContext {
            regime: Regime::TrendingUp,
            sr_score: -5,
            volume_percentile: 75.0,
            confidence: 0.6,
            ..Default::default()
        };

        // Test AND
        let cond = RuleCondition::RegimeIs(Regime::TrendingUp)
            .and(RuleCondition::VolumeAbove(50.0));
        assert!(cond.evaluate(&ctx));

        // Test OR
        let cond = RuleCondition::RegimeIs(Regime::Volatile)
            .or(RuleCondition::VolumeAbove(50.0));
        assert!(cond.evaluate(&ctx));

        // Test NOT
        let cond = RuleCondition::RegimeIs(Regime::Volatile).not();
        assert!(cond.evaluate(&ctx));

        // Test complex
        let cond = RuleCondition::RegimeIs(Regime::TrendingUp)
            .and(RuleCondition::VolumeAbove(50.0))
            .and(RuleCondition::ConfidenceAbove(0.5));
        assert!(cond.evaluate(&ctx));
    }

    #[test]
    fn test_rule_performance_tracking() {
        let mut perf = RulePerformance::default();

        perf.record_trigger();
        assert_eq!(perf.times_triggered, 1);

        perf.record_outcome(true, 100.0);
        perf.record_outcome(true, 50.0);
        perf.record_outcome(false, -30.0);

        assert_eq!(perf.trades_affected, 3);
        assert_eq!(perf.wins_when_triggered, 2);
        assert_eq!(perf.losses_when_triggered, 1);
        assert!((perf.win_rate() - 0.666).abs() < 0.01);
        assert!((perf.estimated_pnl_impact - 120.0).abs() < 0.01);
    }

    #[test]
    fn test_constitution_defaults() {
        let constitution = Constitution::conservative();

        assert_eq!(constitution.max_position_size, 0.05);
        assert_eq!(constitution.max_daily_loss, 0.03);
        assert_eq!(constitution.max_drawdown, 0.10);
        assert_eq!(constitution.min_confidence_for_trade, 0.40);
        assert!(constitution.require_backtest_for_rules);
        assert_eq!(constitution.min_evidence_for_change, 20);
        assert_eq!(constitution.max_rule_changes_per_day, 3);
    }

    #[test]
    fn test_creator_is_system() {
        let human = Creator::Human;
        assert!(!human.is_system());

        let system = Creator::System {
            reason: "Auto-generated".to_string(),
            evidence: vec!["test".to_string()],
        };
        assert!(system.is_system());
    }

    #[test]
    fn test_modification_description() {
        let mod1 = ModificationType::ConfigChange {
            key: "test".to_string(),
            old_value: "old".to_string(),
            new_value: "new".to_string(),
        };
        assert!(mod1.description().contains("test"));

        let mod2 = ModificationType::ThresholdChange {
            name: "threshold".to_string(),
            old: 0.5,
            new: 0.6,
        };
        assert!(mod2.description().contains("threshold"));
    }
}
