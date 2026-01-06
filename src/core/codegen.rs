//! Code Self-Modification with Safety Sandbox
//!
//! This module enables the trading system to generate and deploy new code
//! at runtime, with strict safety constraints enforced by a sandbox.
//!
//! Since Rust cannot compile code at runtime, we use an Expression tree
//! approach that can be evaluated dynamically while maintaining type safety.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

use super::regime::Regime;
use super::selfmod::ConstitutionalGuard;
use super::weakness::{Weakness, WeaknessType};
use super::counterfactual::{TradingInsight, InsightType, CounterfactualResult};
use super::consolidation::Pattern;

// ==================== Core Types ====================

/// Type of generated code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeType {
    /// New filter condition for signals
    SignalFilter,
    /// Modifies confidence calculation
    ConfidenceAdjuster,
    /// New exit condition
    ExitRule,
    /// New feature for ML models
    FeatureExtractor,
    /// Modifies position sizing
    RiskAdjuster,
}

impl std::fmt::Display for CodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeType::SignalFilter => write!(f, "SignalFilter"),
            CodeType::ConfidenceAdjuster => write!(f, "ConfidenceAdjuster"),
            CodeType::ExitRule => write!(f, "ExitRule"),
            CodeType::FeatureExtractor => write!(f, "FeatureExtractor"),
            CodeType::RiskAdjuster => write!(f, "RiskAdjuster"),
        }
    }
}

/// Results from backtesting generated code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    /// Number of trades simulated
    pub trades_simulated: u32,
    /// Change in win rate (positive = improvement)
    pub win_rate_change: f64,
    /// Change in total PnL
    pub pnl_change: f64,
    /// Change in Sharpe ratio
    pub sharpe_change: f64,
    /// Change in max drawdown (negative = improvement)
    pub max_dd_change: f64,
}

impl BacktestResults {
    /// Check if results are positive overall
    pub fn is_improvement(&self) -> bool {
        self.win_rate_change > 0.0 ||
        self.pnl_change > 0.0 ||
        self.sharpe_change > 0.0
    }

    /// Get overall score
    pub fn score(&self) -> f64 {
        // Weighted score of improvements
        self.win_rate_change * 100.0 +  // Win rate weighted heavily
        self.pnl_change * 0.01 +         // PnL in dollars
        self.sharpe_change * 10.0 +      // Sharpe improvement
        self.max_dd_change * -50.0       // DD reduction (negative is good)
    }
}

/// Results from testing generated code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    /// Whether all tests passed
    pub passed: bool,
    /// Number of tests run
    pub tests_run: u32,
    /// Number of tests that passed
    pub tests_passed: u32,
    /// Whether code compiled successfully
    pub compile_success: bool,
    /// Whether sandbox validation passed
    pub sandbox_success: bool,
    /// Backtest results if available
    pub backtest_results: Option<BacktestResults>,
}

impl TestResults {
    /// Create successful test results
    pub fn success(tests_run: u32) -> Self {
        Self {
            passed: true,
            tests_run,
            tests_passed: tests_run,
            compile_success: true,
            sandbox_success: true,
            backtest_results: None,
        }
    }

    /// Create failed test results
    pub fn failure(tests_run: u32, tests_passed: u32, reason: &str) -> Self {
        info!("Test failure: {}", reason);
        Self {
            passed: false,
            tests_run,
            tests_passed,
            compile_success: true,
            sandbox_success: true,
            backtest_results: None,
        }
    }
}

/// Performance metrics for deployed code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePerformance {
    /// How many times the code was executed
    pub times_executed: u32,
    /// Number of trades affected by this code
    pub trades_affected: u32,
    /// Actual PnL impact observed
    pub actual_pnl_impact: f64,
    /// When the code was deployed
    pub deployed_at: DateTime<Utc>,
}

impl CodePerformance {
    /// Create new performance tracker
    pub fn new() -> Self {
        Self {
            times_executed: 0,
            trades_affected: 0,
            actual_pnl_impact: 0.0,
            deployed_at: Utc::now(),
        }
    }

    /// Record an execution
    pub fn record_execution(&mut self, affected_trade: bool, pnl_impact: f64) {
        self.times_executed += 1;
        if affected_trade {
            self.trades_affected += 1;
            self.actual_pnl_impact += pnl_impact;
        }
    }
}

impl Default for CodePerformance {
    fn default() -> Self {
        Self::new()
    }
}

/// Generated code unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCode {
    /// Unique identifier
    pub id: u64,
    /// Type of code
    pub code_type: CodeType,
    /// Human-readable source representation
    pub source_code: String,
    /// Description of what this code does
    pub description: String,
    /// When the code was generated
    pub generated_at: DateTime<Utc>,
    /// Test results
    pub test_results: Option<TestResults>,
    /// Whether currently deployed
    pub deployed: bool,
    /// Performance metrics if deployed
    pub performance: Option<CodePerformance>,
    /// The actual executable expression
    pub expression: Expression,
}

impl GeneratedCode {
    /// Create new generated code
    pub fn new(
        id: u64,
        code_type: CodeType,
        source_code: String,
        description: String,
        expression: Expression,
    ) -> Self {
        Self {
            id,
            code_type,
            source_code,
            description,
            generated_at: Utc::now(),
            test_results: None,
            deployed: false,
            performance: None,
            expression,
        }
    }

    /// Mark as deployed
    pub fn deploy(&mut self) {
        self.deployed = true;
        self.performance = Some(CodePerformance::new());
    }

    /// Mark as undeployed
    pub fn undeploy(&mut self) {
        self.deployed = false;
    }
}

// ==================== Expression System ====================

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    /// Apply the comparison
    pub fn apply(&self, left: f64, right: f64) -> bool {
        match self {
            CompareOp::Eq => (left - right).abs() < f64::EPSILON,
            CompareOp::Ne => (left - right).abs() >= f64::EPSILON,
            CompareOp::Lt => left < right,
            CompareOp::Le => left <= right,
            CompareOp::Gt => left > right,
            CompareOp::Ge => left >= right,
        }
    }
}

impl std::fmt::Display for CompareOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompareOp::Eq => write!(f, "=="),
            CompareOp::Ne => write!(f, "!="),
            CompareOp::Lt => write!(f, "<"),
            CompareOp::Le => write!(f, "<="),
            CompareOp::Gt => write!(f, ">"),
            CompareOp::Ge => write!(f, ">="),
        }
    }
}

/// Expression tree for runtime evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    /// Literal numeric value
    Literal(f64),
    /// Variable reference by name
    Variable(String),
    /// Regime check
    RegimeIs(Regime),
    /// Regime is not check
    RegimeIsNot(Regime),
    /// Comparison expression
    Compare {
        left: Box<Expression>,
        op: CompareOp,
        right: Box<Expression>,
    },
    /// Logical AND
    And(Box<Expression>, Box<Expression>),
    /// Logical OR
    Or(Box<Expression>, Box<Expression>),
    /// Logical NOT
    Not(Box<Expression>),
    /// Conditional expression
    IfThenElse {
        cond: Box<Expression>,
        then_val: Box<Expression>,
        else_val: Box<Expression>,
    },
    /// Addition
    Add(Box<Expression>, Box<Expression>),
    /// Subtraction
    Sub(Box<Expression>, Box<Expression>),
    /// Multiplication
    Mul(Box<Expression>, Box<Expression>),
    /// Division
    Div(Box<Expression>, Box<Expression>),
    /// Minimum of two values
    Min(Box<Expression>, Box<Expression>),
    /// Maximum of two values
    Max(Box<Expression>, Box<Expression>),
    /// Clamp value between min and max
    Clamp {
        value: Box<Expression>,
        min: Box<Expression>,
        max: Box<Expression>,
    },
    /// Absolute value
    Abs(Box<Expression>),
    /// Always true (pass-through filter)
    True,
    /// Always false (block all)
    False,
}

impl Expression {
    /// Create a simple variable reference
    pub fn var(name: &str) -> Self {
        Expression::Variable(name.to_string())
    }

    /// Create a literal value
    pub fn lit(value: f64) -> Self {
        Expression::Literal(value)
    }

    /// Create a comparison
    pub fn compare(left: Expression, op: CompareOp, right: Expression) -> Self {
        Expression::Compare {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    /// Create AND expression
    pub fn and(left: Expression, right: Expression) -> Self {
        Expression::And(Box::new(left), Box::new(right))
    }

    /// Create OR expression
    pub fn or(left: Expression, right: Expression) -> Self {
        Expression::Or(Box::new(left), Box::new(right))
    }

    /// Create NOT expression
    pub fn not(expr: Expression) -> Self {
        Expression::Not(Box::new(expr))
    }

    /// Create if-then-else
    pub fn if_then_else(cond: Expression, then_val: Expression, else_val: Expression) -> Self {
        Expression::IfThenElse {
            cond: Box::new(cond),
            then_val: Box::new(then_val),
            else_val: Box::new(else_val),
        }
    }

    /// Create addition
    pub fn add(left: Expression, right: Expression) -> Self {
        Expression::Add(Box::new(left), Box::new(right))
    }

    /// Create multiplication
    pub fn mul(left: Expression, right: Expression) -> Self {
        Expression::Mul(Box::new(left), Box::new(right))
    }

    /// Create division with safety check
    pub fn div(left: Expression, right: Expression) -> Self {
        Expression::Div(Box::new(left), Box::new(right))
    }

    /// Create clamp expression
    pub fn clamp(value: Expression, min: Expression, max: Expression) -> Self {
        Expression::Clamp {
            value: Box::new(value),
            min: Box::new(min),
            max: Box::new(max),
        }
    }

    /// Get human-readable representation
    pub fn to_source(&self) -> String {
        match self {
            Expression::Literal(v) => format!("{:.4}", v),
            Expression::Variable(name) => format!("ctx.{}", name),
            Expression::RegimeIs(r) => format!("ctx.regime == {:?}", r),
            Expression::RegimeIsNot(r) => format!("ctx.regime != {:?}", r),
            Expression::Compare { left, op, right } => {
                format!("({} {} {})", left.to_source(), op, right.to_source())
            }
            Expression::And(l, r) => format!("({} && {})", l.to_source(), r.to_source()),
            Expression::Or(l, r) => format!("({} || {})", l.to_source(), r.to_source()),
            Expression::Not(e) => format!("!{}", e.to_source()),
            Expression::IfThenElse { cond, then_val, else_val } => {
                format!(
                    "if {} {{ {} }} else {{ {} }}",
                    cond.to_source(),
                    then_val.to_source(),
                    else_val.to_source()
                )
            }
            Expression::Add(l, r) => format!("({} + {})", l.to_source(), r.to_source()),
            Expression::Sub(l, r) => format!("({} - {})", l.to_source(), r.to_source()),
            Expression::Mul(l, r) => format!("({} * {})", l.to_source(), r.to_source()),
            Expression::Div(l, r) => format!("({} / {})", l.to_source(), r.to_source()),
            Expression::Min(l, r) => format!("min({}, {})", l.to_source(), r.to_source()),
            Expression::Max(l, r) => format!("max({}, {})", l.to_source(), r.to_source()),
            Expression::Clamp { value, min, max } => {
                format!("clamp({}, {}, {})", value.to_source(), min.to_source(), max.to_source())
            }
            Expression::Abs(e) => format!("abs({})", e.to_source()),
            Expression::True => "true".to_string(),
            Expression::False => "false".to_string(),
        }
    }
}

/// Context for evaluating expressions
#[derive(Debug, Clone)]
pub struct EvalContext {
    /// Variable values
    pub variables: HashMap<String, f64>,
    /// Current regime
    pub regime: Regime,
}

impl EvalContext {
    /// Create new evaluation context
    pub fn new(regime: Regime) -> Self {
        Self {
            variables: HashMap::new(),
            regime,
        }
    }

    /// Set a variable value
    pub fn set(&mut self, name: &str, value: f64) {
        self.variables.insert(name.to_string(), value);
    }

    /// Get a variable value
    pub fn get(&self, name: &str) -> f64 {
        self.variables.get(name).copied().unwrap_or(0.0)
    }

    /// Create from trade context values
    pub fn from_trade_context(
        sr_score: i32,
        volume_percentile: f64,
        confidence: f64,
        atr_pct: f64,
        distance_to_sr_pct: f64,
        is_long: bool,
        current_pnl: f64,
        bars_held: u32,
        regime: Regime,
    ) -> Self {
        let mut ctx = Self::new(regime);
        ctx.set("sr_score", sr_score as f64);
        ctx.set("volume_percentile", volume_percentile);
        ctx.set("volume_pct", volume_percentile); // Alias
        ctx.set("confidence", confidence);
        ctx.set("atr_pct", atr_pct);
        ctx.set("distance_to_sr_pct", distance_to_sr_pct);
        ctx.set("is_long", if is_long { 1.0 } else { 0.0 });
        ctx.set("current_pnl", current_pnl);
        ctx.set("bars_held", bars_held as f64);
        ctx
    }
}

/// Evaluate an expression in a context
pub fn evaluate(expr: &Expression, ctx: &EvalContext) -> f64 {
    match expr {
        Expression::Literal(v) => *v,
        Expression::Variable(name) => ctx.get(name),
        Expression::RegimeIs(r) => if ctx.regime == *r { 1.0 } else { 0.0 },
        Expression::RegimeIsNot(r) => if ctx.regime != *r { 1.0 } else { 0.0 },
        Expression::Compare { left, op, right } => {
            let l = evaluate(left, ctx);
            let r = evaluate(right, ctx);
            if op.apply(l, r) { 1.0 } else { 0.0 }
        }
        Expression::And(l, r) => {
            let lv = evaluate(l, ctx);
            let rv = evaluate(r, ctx);
            if lv > 0.5 && rv > 0.5 { 1.0 } else { 0.0 }
        }
        Expression::Or(l, r) => {
            let lv = evaluate(l, ctx);
            let rv = evaluate(r, ctx);
            if lv > 0.5 || rv > 0.5 { 1.0 } else { 0.0 }
        }
        Expression::Not(e) => {
            let v = evaluate(e, ctx);
            if v > 0.5 { 0.0 } else { 1.0 }
        }
        Expression::IfThenElse { cond, then_val, else_val } => {
            let c = evaluate(cond, ctx);
            if c > 0.5 {
                evaluate(then_val, ctx)
            } else {
                evaluate(else_val, ctx)
            }
        }
        Expression::Add(l, r) => evaluate(l, ctx) + evaluate(r, ctx),
        Expression::Sub(l, r) => evaluate(l, ctx) - evaluate(r, ctx),
        Expression::Mul(l, r) => evaluate(l, ctx) * evaluate(r, ctx),
        Expression::Div(l, r) => {
            let rv = evaluate(r, ctx);
            if rv.abs() < f64::EPSILON {
                0.0 // Safe division by zero
            } else {
                evaluate(l, ctx) / rv
            }
        }
        Expression::Min(l, r) => evaluate(l, ctx).min(evaluate(r, ctx)),
        Expression::Max(l, r) => evaluate(l, ctx).max(evaluate(r, ctx)),
        Expression::Clamp { value, min, max } => {
            let v = evaluate(value, ctx);
            let mn = evaluate(min, ctx);
            let mx = evaluate(max, ctx);
            v.clamp(mn, mx)
        }
        Expression::Abs(e) => evaluate(e, ctx).abs(),
        Expression::True => 1.0,
        Expression::False => 0.0,
    }
}

/// Evaluate expression as boolean
pub fn evaluate_bool(expr: &Expression, ctx: &EvalContext) -> bool {
    evaluate(expr, ctx) > 0.5
}

// ==================== Safety Sandbox ====================

/// Safety violation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyViolation {
    /// Unsafe block detected
    UnsafeBlock,
    /// Forbidden import used
    ForbiddenImport(String),
    /// File system access attempted
    FileSystemAccess,
    /// Network access attempted
    NetworkAccess,
    /// External command execution attempted
    ExternalCommand,
    /// Potential infinite loop detected
    PotentialInfiniteLoop,
    /// Excessive recursion depth
    ExcessiveRecursion,
    /// Expression too complex
    ExpressionTooComplex,
}

impl std::fmt::Display for SafetyViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafetyViolation::UnsafeBlock => write!(f, "Unsafe block detected"),
            SafetyViolation::ForbiddenImport(s) => write!(f, "Forbidden import: {}", s),
            SafetyViolation::FileSystemAccess => write!(f, "File system access attempted"),
            SafetyViolation::NetworkAccess => write!(f, "Network access attempted"),
            SafetyViolation::ExternalCommand => write!(f, "External command execution"),
            SafetyViolation::PotentialInfiniteLoop => write!(f, "Potential infinite loop"),
            SafetyViolation::ExcessiveRecursion => write!(f, "Excessive recursion depth"),
            SafetyViolation::ExpressionTooComplex => write!(f, "Expression too complex"),
        }
    }
}

/// Safety sandbox for validating generated code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sandbox {
    /// Path for temporary files (unused in expression mode)
    pub path: String,
    /// Maximum expression depth
    pub max_expression_depth: u32,
    /// Maximum execution steps
    pub max_execution_steps: u64,
}

impl Sandbox {
    /// Create new sandbox
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            max_expression_depth: 20,
            max_execution_steps: 10000,
        }
    }

    /// Validate expression safety
    pub fn validate_safety(&self, expr: &Expression) -> Result<(), SafetyViolation> {
        // Check expression depth
        let depth = self.expression_depth(expr);
        if depth > self.max_expression_depth {
            return Err(SafetyViolation::ExpressionTooComplex);
        }

        // Check for potential infinite recursion in expression structure
        // (Our expressions don't support user-defined recursion, so this is always safe)

        Ok(())
    }

    /// Calculate expression depth
    fn expression_depth(&self, expr: &Expression) -> u32 {
        match expr {
            Expression::Literal(_) | Expression::Variable(_) |
            Expression::RegimeIs(_) | Expression::RegimeIsNot(_) |
            Expression::True | Expression::False => 1,

            Expression::Compare { left, right, .. } |
            Expression::And(left, right) |
            Expression::Or(left, right) |
            Expression::Add(left, right) |
            Expression::Sub(left, right) |
            Expression::Mul(left, right) |
            Expression::Div(left, right) |
            Expression::Min(left, right) |
            Expression::Max(left, right) => {
                1 + self.expression_depth(left).max(self.expression_depth(right))
            }

            Expression::Not(e) | Expression::Abs(e) => 1 + self.expression_depth(e),

            Expression::IfThenElse { cond, then_val, else_val } => {
                1 + self.expression_depth(cond)
                    .max(self.expression_depth(then_val))
                    .max(self.expression_depth(else_val))
            }

            Expression::Clamp { value, min, max } => {
                1 + self.expression_depth(value)
                    .max(self.expression_depth(min))
                    .max(self.expression_depth(max))
            }
        }
    }

    /// Run tests on generated code
    pub fn test(&self, code: &GeneratedCode, test_cases: &[EvalContext]) -> TestResults {
        // Validate safety first
        if let Err(violation) = self.validate_safety(&code.expression) {
            return TestResults {
                passed: false,
                tests_run: 0,
                tests_passed: 0,
                compile_success: true,
                sandbox_success: false,
                backtest_results: None,
            };
        }

        let mut tests_passed = 0;
        let tests_run = test_cases.len() as u32;

        for ctx in test_cases {
            // Run expression and check it doesn't panic
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                evaluate(&code.expression, ctx)
            }));

            if result.is_ok() {
                tests_passed += 1;
            }
        }

        TestResults {
            passed: tests_passed == tests_run,
            tests_run,
            tests_passed,
            compile_success: true,
            sandbox_success: true,
            backtest_results: None,
        }
    }
}

impl Default for Sandbox {
    fn default() -> Self {
        Self::new("/tmp/sovereign_sandbox")
    }
}

// ==================== Code Generator ====================

/// Code templates for different types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeTemplate {
    /// Signal filter template
    SignalFilterTemplate {
        condition: String,
        description: String,
    },
    /// Confidence adjuster template
    ConfidenceAdjusterTemplate {
        adjustment_logic: String,
        triggers: Vec<String>,
    },
    /// Exit rule template
    ExitRuleTemplate {
        exit_condition: String,
        description: String,
    },
}

/// Code generator that creates new code from patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGenerator {
    /// Base templates by code type
    #[serde(skip)]
    templates: HashMap<CodeType, Vec<String>>,
    /// History of generated code
    generated_history: Vec<GeneratedCode>,
    /// Currently active generated code
    active_code: HashMap<u64, GeneratedCode>,
    /// Sandbox path
    sandbox_path: String,
    /// Maximum active generated code pieces
    max_active_generated: usize,
    /// Next code ID
    next_id: u64,
}

impl CodeGenerator {
    /// Create new code generator
    pub fn new(sandbox_path: &str) -> Self {
        Self {
            templates: HashMap::new(),
            generated_history: Vec::new(),
            active_code: HashMap::new(),
            sandbox_path: sandbox_path.to_string(),
            max_active_generated: 10,
            next_id: 1,
        }
    }

    /// Get next ID
    fn next_code_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Generate signal filter from weakness
    pub fn generate_signal_filter(&mut self, weakness: &Weakness) -> GeneratedCode {
        let id = self.next_code_id();

        let (expression, description) = match &weakness.weakness_type {
            WeaknessType::RegimeWeakness { regime, win_rate, .. } => {
                // Skip trades in weak regime
                let expr = Expression::RegimeIsNot(regime.clone());
                let desc = format!(
                    "Filter: Skip {} regime (win rate: {:.1}%)",
                    regime, win_rate * 100.0
                );
                (expr, desc)
            }
            WeaknessType::SRScoreWeakness { score_range, win_rate, .. } => {
                // Require S/R score above threshold
                let (low, _high) = score_range;
                let expr = Expression::compare(
                    Expression::var("sr_score"),
                    CompareOp::Ge,
                    Expression::lit(*low as f64),
                );
                let desc = format!(
                    "Filter: Require S/R score >= {} (weak range: {:.1}% win rate)",
                    low, win_rate * 100.0
                );
                (expr, desc)
            }
            WeaknessType::VolumeThresholdWeakness { threshold_range, win_rate, .. } => {
                let (low, high) = threshold_range;
                // Avoid the weak volume range
                let expr = Expression::or(
                    Expression::compare(
                        Expression::var("volume_pct"),
                        CompareOp::Lt,
                        Expression::lit(*low),
                    ),
                    Expression::compare(
                        Expression::var("volume_pct"),
                        CompareOp::Gt,
                        Expression::lit(*high),
                    ),
                );
                let desc = format!(
                    "Filter: Avoid volume {:.0}-{:.0}% range (win rate: {:.1}%)",
                    low, high, win_rate * 100.0
                );
                (expr, desc)
            }
            WeaknessType::TimeOfDayWeakness { hour_utc, win_rate, .. } => {
                // Note: We can't easily check time in expression, use pass-through
                let expr = Expression::True;
                let desc = format!(
                    "Filter: Weak at hour {} UTC ({:.1}% win rate) - logging only",
                    hour_utc, win_rate * 100.0
                );
                (expr, desc)
            }
            WeaknessType::SymbolWeakness { symbol, win_rate, .. } => {
                // Pass-through with logging
                let expr = Expression::True;
                let desc = format!(
                    "Filter: Symbol {} weak ({:.1}% win rate) - logging only",
                    symbol, win_rate * 100.0
                );
                (expr, desc)
            }
            WeaknessType::ClusterWeakness { cluster, win_rate, .. } => {
                let expr = Expression::True;
                let desc = format!(
                    "Filter: Cluster {:?} weak ({:.1}% win rate) - logging only",
                    cluster, win_rate * 100.0
                );
                (expr, desc)
            }
        };

        let source_code = format!(
            "fn filter_{}(ctx: &TradeContext) -> bool {{\n    {}\n}}",
            id,
            expression.to_source()
        );

        let mut code = GeneratedCode::new(
            id,
            CodeType::SignalFilter,
            source_code,
            description,
            expression,
        );

        self.generated_history.push(code.clone());
        code
    }

    /// Generate confidence adjuster from insight
    pub fn generate_confidence_adjuster(&mut self, insight: &TradingInsight) -> GeneratedCode {
        let id = self.next_code_id();

        let (expression, description) = match &insight.insight_type {
            InsightType::ExitTooEarly => {
                // In trending regimes, boost confidence to hold longer
                let expr = Expression::if_then_else(
                    Expression::or(
                        Expression::RegimeIs(Regime::TrendingUp),
                        Expression::RegimeIs(Regime::TrendingDown),
                    ),
                    Expression::mul(
                        Expression::var("confidence"),
                        Expression::lit(1.1), // 10% boost
                    ),
                    Expression::var("confidence"),
                );
                let desc = "Adjuster: Boost confidence in trending regimes (exit too early pattern)".to_string();
                (expr, desc)
            }
            InsightType::ExitTooLate => {
                // Reduce confidence to exit earlier
                let expr = Expression::mul(
                    Expression::var("confidence"),
                    Expression::lit(0.9), // 10% reduction
                );
                let desc = "Adjuster: Reduce confidence globally (exit too late pattern)".to_string();
                (expr, desc)
            }
            InsightType::SizeTooSmall => {
                // Boost confidence for higher conviction
                let expr = Expression::if_then_else(
                    Expression::compare(
                        Expression::var("confidence"),
                        CompareOp::Gt,
                        Expression::lit(0.6),
                    ),
                    Expression::mul(
                        Expression::var("confidence"),
                        Expression::lit(1.15),
                    ),
                    Expression::var("confidence"),
                );
                let desc = "Adjuster: Boost high-confidence trades (size too small pattern)".to_string();
                (expr, desc)
            }
            InsightType::SizeTooLarge => {
                // Reduce confidence for lower conviction
                let expr = Expression::mul(
                    Expression::var("confidence"),
                    Expression::lit(0.85),
                );
                let desc = "Adjuster: Reduce confidence globally (size too large pattern)".to_string();
                (expr, desc)
            }
            InsightType::WrongDirection => {
                // Be more cautious overall
                let expr = Expression::mul(
                    Expression::var("confidence"),
                    Expression::lit(0.9),
                );
                let desc = "Adjuster: Reduce confidence (wrong direction pattern)".to_string();
                (expr, desc)
            }
            InsightType::ShouldHaveSkipped => {
                // Require higher confidence
                let expr = Expression::if_then_else(
                    Expression::compare(
                        Expression::var("confidence"),
                        CompareOp::Lt,
                        Expression::lit(0.55),
                    ),
                    Expression::lit(0.0), // Block low confidence
                    Expression::var("confidence"),
                );
                let desc = "Adjuster: Block low confidence trades (should have skipped pattern)".to_string();
                (expr, desc)
            }
            InsightType::GoodDecision => {
                // No adjustment needed
                let expr = Expression::var("confidence");
                let desc = "Adjuster: No change (good decisions confirmed)".to_string();
                (expr, desc)
            }
        };

        // Clamp final result to valid range
        let clamped_expr = Expression::clamp(
            expression,
            Expression::lit(0.0),
            Expression::lit(1.0),
        );

        let source_code = format!(
            "fn adjust_confidence_{}(base: f64, ctx: &TradeContext) -> f64 {{\n    {}\n}}",
            id,
            clamped_expr.to_source()
        );

        let mut code = GeneratedCode::new(
            id,
            CodeType::ConfidenceAdjuster,
            source_code,
            description,
            clamped_expr,
        );

        self.generated_history.push(code.clone());
        code
    }

    /// Generate exit rule from counterfactual analysis
    pub fn generate_exit_rule(&mut self, cf_result: &CounterfactualResult) -> GeneratedCode {
        let id = self.next_code_id();

        // Generate exit condition based on counterfactual analysis
        // If earlier exit would have been better, create tighter exit rules
        let (expression, description) = if cf_result.counterfactual_pnl > cf_result.actual_pnl {
            // Earlier exit would have been better
            let pnl_threshold = cf_result.actual_pnl.abs() * 0.5;

            let expr = Expression::or(
                // Exit if profit drops significantly
                Expression::and(
                    Expression::compare(
                        Expression::var("current_pnl"),
                        CompareOp::Gt,
                        Expression::lit(0.0),
                    ),
                    Expression::compare(
                        Expression::var("bars_held"),
                        CompareOp::Gt,
                        Expression::lit(10.0),
                    ),
                ),
                // Exit if in loss too long
                Expression::and(
                    Expression::compare(
                        Expression::var("current_pnl"),
                        CompareOp::Lt,
                        Expression::lit(-pnl_threshold),
                    ),
                    Expression::compare(
                        Expression::var("bars_held"),
                        CompareOp::Gt,
                        Expression::lit(5.0),
                    ),
                ),
            );

            let desc = format!(
                "Exit: Tighter exit rules (counterfactual: ${:.2} better with earlier exit)",
                cf_result.counterfactual_pnl - cf_result.actual_pnl
            );
            (expr, desc)
        } else {
            // Later exit would have been better - use trailing stop
            let expr = Expression::and(
                Expression::compare(
                    Expression::var("current_pnl"),
                    CompareOp::Lt,
                    Expression::lit(-50.0), // $50 loss threshold
                ),
                Expression::compare(
                    Expression::var("bars_held"),
                    CompareOp::Gt,
                    Expression::lit(20.0),
                ),
            );

            let desc = format!(
                "Exit: Extended hold with stop (counterfactual: ${:.2} better with later exit)",
                cf_result.counterfactual_pnl - cf_result.actual_pnl
            );
            (expr, desc)
        };

        let source_code = format!(
            "fn check_exit_{}(ctx: &TradeContext, pnl: f64) -> bool {{\n    {}\n}}",
            id,
            expression.to_source()
        );

        let mut code = GeneratedCode::new(
            id,
            CodeType::ExitRule,
            source_code,
            description,
            expression,
        );

        self.generated_history.push(code.clone());
        code
    }

    /// Generate feature extractor from discovered pattern
    pub fn generate_feature_extractor(&mut self, pattern: &Pattern) -> GeneratedCode {
        let id = self.next_code_id();

        // Create expression that computes similarity to pattern centroid
        let mut terms = Vec::new();

        // Add terms for each feature in the pattern
        for (i, &centroid_val) in pattern.centroid.iter().enumerate() {
            let var_name = match i {
                0 => "sr_score",
                1 => "volume_pct",
                2 => "confidence",
                3 => "atr_pct",
                _ => continue,
            };

            // Compute squared distance from centroid
            let term = Expression::Mul(
                Box::new(Expression::Sub(
                    Box::new(Expression::var(var_name)),
                    Box::new(Expression::lit(centroid_val)),
                )),
                Box::new(Expression::Sub(
                    Box::new(Expression::var(var_name)),
                    Box::new(Expression::lit(centroid_val)),
                )),
            );
            terms.push(term);
        }

        // Sum squared distances and convert to similarity (1 / (1 + dist))
        let sum_expr = if terms.is_empty() {
            Expression::lit(0.0)
        } else {
            let mut sum = terms.remove(0);
            for term in terms {
                sum = Expression::add(sum, term);
            }
            sum
        };

        let similarity_expr = Expression::div(
            Expression::lit(1.0),
            Expression::add(Expression::lit(1.0), sum_expr),
        );

        let source_code = format!(
            "fn feature_pattern_similarity_{}(ctx: &TradeContext) -> f64 {{\n    {}\n}}",
            id,
            similarity_expr.to_source()
        );

        let description = format!(
            "Feature: Similarity to pattern '{}' (win rate: {:.1}%, {} examples)",
            pattern.name,
            pattern.win_rate * 100.0,
            pattern.member_count
        );

        let mut code = GeneratedCode::new(
            id,
            CodeType::FeatureExtractor,
            source_code,
            description,
            similarity_expr,
        );

        self.generated_history.push(code.clone());
        code
    }

    /// Get history of generated code
    pub fn get_history(&self) -> &[GeneratedCode] {
        &self.generated_history
    }

    /// Get active code count
    pub fn active_count(&self) -> usize {
        self.active_code.len()
    }

    /// Get history count
    pub fn history_count(&self) -> usize {
        self.generated_history.len()
    }
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new("/tmp/sovereign_sandbox")
    }
}

// ==================== Deployment System ====================

/// Deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStatus {
    /// Code is active
    Active,
    /// Code was rolled back
    RolledBack { reason: String },
    /// Code was superseded by newer version
    Superseded { by: u64 },
}

/// Record of a deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentRecord {
    /// Code ID
    pub code_id: u64,
    /// When deployed
    pub deployed_at: DateTime<Utc>,
    /// Current status
    pub status: DeploymentStatus,
    /// When rolled back (if applicable)
    pub rollback_at: Option<DateTime<Utc>>,
}

/// Error during code proposal
#[derive(Debug, Clone)]
pub enum ProposalError {
    /// Safety violation
    SafetyViolation(SafetyViolation),
    /// Test failure
    TestFailure(String),
    /// Too many active code pieces
    TooManyActive,
    /// Guard rejected
    GuardRejected(String),
}

impl std::fmt::Display for ProposalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProposalError::SafetyViolation(v) => write!(f, "Safety violation: {}", v),
            ProposalError::TestFailure(s) => write!(f, "Test failure: {}", s),
            ProposalError::TooManyActive => write!(f, "Too many active code pieces"),
            ProposalError::GuardRejected(s) => write!(f, "Guard rejected: {}", s),
        }
    }
}

/// Error during deployment
#[derive(Debug, Clone)]
pub enum DeployError {
    /// Code not found
    NotFound,
    /// Not approved
    NotApproved,
    /// Already deployed
    AlreadyDeployed,
}

/// Error during rollback
#[derive(Debug, Clone)]
pub enum RollbackError {
    /// Code not found
    NotFound,
    /// Not deployed
    NotDeployed,
}

/// Code deployer manages deployment of generated code
#[derive(Debug, Serialize, Deserialize)]
pub struct CodeDeployer {
    /// Code generator
    generator: CodeGenerator,
    /// Safety sandbox
    sandbox: Sandbox,
    /// Deployment history
    deployment_history: Vec<DeploymentRecord>,
    /// Rollback stack
    rollback_stack: Vec<(u64, GeneratedCode)>,
    /// Pending code awaiting approval
    pending_code: HashMap<u64, GeneratedCode>,
    /// Active deployed code
    active_code: HashMap<u64, GeneratedCode>,
    /// Maximum active code pieces
    max_active: usize,
}

impl CodeDeployer {
    /// Create new code deployer
    pub fn new(sandbox_path: &str) -> Self {
        Self {
            generator: CodeGenerator::new(sandbox_path),
            sandbox: Sandbox::new(sandbox_path),
            deployment_history: Vec::new(),
            rollback_stack: Vec::new(),
            pending_code: HashMap::new(),
            active_code: HashMap::new(),
            max_active: 10,
        }
    }

    /// Propose new code
    pub fn propose_code(&mut self, mut code: GeneratedCode) -> Result<u64, ProposalError> {
        // Validate safety
        self.sandbox.validate_safety(&code.expression)
            .map_err(ProposalError::SafetyViolation)?;

        // Run tests
        let test_cases = self.generate_test_cases();
        let results = self.sandbox.test(&code, &test_cases);

        if !results.passed {
            return Err(ProposalError::TestFailure(
                format!("{}/{} tests passed", results.tests_passed, results.tests_run)
            ));
        }

        code.test_results = Some(results);

        // Check capacity
        if self.active_code.len() >= self.max_active {
            return Err(ProposalError::TooManyActive);
        }

        let id = code.id;
        self.pending_code.insert(id, code);

        info!("[CODEGEN] Proposed code {}: passed safety and tests", id);
        Ok(id)
    }

    /// Generate test cases for validation
    fn generate_test_cases(&self) -> Vec<EvalContext> {
        let mut cases = Vec::new();

        // Test various regime combinations
        for regime in [Regime::Volatile, Regime::TrendingUp, Regime::TrendingDown, Regime::Ranging] {
            let mut ctx = EvalContext::new(regime);
            ctx.set("sr_score", -5.0);
            ctx.set("volume_pct", 75.0);
            ctx.set("confidence", 0.65);
            ctx.set("atr_pct", 2.0);
            ctx.set("distance_to_sr_pct", 0.5);
            ctx.set("is_long", 1.0);
            ctx.set("current_pnl", 50.0);
            ctx.set("bars_held", 5.0);
            cases.push(ctx);
        }

        // Test edge cases
        let mut edge = EvalContext::new(Regime::Ranging);
        edge.set("sr_score", 0.0);
        edge.set("volume_pct", 0.0);
        edge.set("confidence", 0.0);
        cases.push(edge);

        let mut edge2 = EvalContext::new(Regime::Volatile);
        edge2.set("sr_score", -10.0);
        edge2.set("volume_pct", 100.0);
        edge2.set("confidence", 1.0);
        cases.push(edge2);

        cases
    }

    /// Deploy approved code
    pub fn deploy(&mut self, code_id: u64) -> Result<(), DeployError> {
        let mut code = self.pending_code.remove(&code_id)
            .ok_or(DeployError::NotFound)?;

        // Mark as deployed
        code.deploy();

        // Add to active
        self.active_code.insert(code_id, code.clone());

        // Add to rollback stack
        self.rollback_stack.push((code_id, code.clone()));

        // Record deployment
        self.deployment_history.push(DeploymentRecord {
            code_id,
            deployed_at: Utc::now(),
            status: DeploymentStatus::Active,
            rollback_at: None,
        });

        info!("[CODEGEN] Deployed code {}: {}", code_id, code.description);
        Ok(())
    }

    /// Rollback deployed code
    pub fn rollback(&mut self, code_id: u64) -> Result<(), RollbackError> {
        let code = self.active_code.remove(&code_id)
            .ok_or(RollbackError::NotDeployed)?;

        // Update deployment record
        if let Some(record) = self.deployment_history.iter_mut()
            .find(|r| r.code_id == code_id && matches!(r.status, DeploymentStatus::Active))
        {
            record.status = DeploymentStatus::RolledBack {
                reason: "Manual rollback".to_string()
            };
            record.rollback_at = Some(Utc::now());
        }

        info!("[CODEGEN] Rolled back code {}: {}", code_id, code.description);
        Ok(())
    }

    /// Get active filters
    pub fn get_active_filters(&self) -> Vec<&GeneratedCode> {
        self.active_code.values()
            .filter(|c| c.code_type == CodeType::SignalFilter)
            .collect()
    }

    /// Get active confidence adjusters
    pub fn get_active_adjusters(&self) -> Vec<&GeneratedCode> {
        self.active_code.values()
            .filter(|c| c.code_type == CodeType::ConfidenceAdjuster)
            .collect()
    }

    /// Get active exit rules
    pub fn get_active_exit_rules(&self) -> Vec<&GeneratedCode> {
        self.active_code.values()
            .filter(|c| c.code_type == CodeType::ExitRule)
            .collect()
    }

    /// Execute all active filters
    /// Returns true if trade is allowed (all filters pass)
    pub fn execute_filters(&self, ctx: &EvalContext) -> bool {
        for code in self.get_active_filters() {
            if !evaluate_bool(&code.expression, ctx) {
                info!("[CODEGEN] Filter {} blocked trade", code.id);
                return false;
            }
        }
        true
    }

    /// Execute all active confidence adjusters
    pub fn execute_adjusters(&self, base_confidence: f64, ctx: &EvalContext) -> f64 {
        let mut confidence = base_confidence;
        let mut eval_ctx = ctx.clone();

        for code in self.get_active_adjusters() {
            eval_ctx.set("confidence", confidence);
            confidence = evaluate(&code.expression, &eval_ctx);
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Check exit rules
    /// Returns Some(reason) if exit should trigger
    pub fn check_exit_rules(&self, ctx: &EvalContext) -> Option<String> {
        for code in self.get_active_exit_rules() {
            if evaluate_bool(&code.expression, ctx) {
                return Some(format!("Exit rule {} triggered", code.id));
            }
        }
        None
    }

    /// Get pending code
    pub fn get_pending(&self) -> Vec<&GeneratedCode> {
        self.pending_code.values().collect()
    }

    /// Get all active code
    pub fn get_active(&self) -> Vec<&GeneratedCode> {
        self.active_code.values().collect()
    }

    /// Get generator reference
    pub fn generator(&self) -> &CodeGenerator {
        &self.generator
    }

    /// Get mutable generator reference
    pub fn generator_mut(&mut self) -> &mut CodeGenerator {
        &mut self.generator
    }

    /// Get pending count
    pub fn pending_count(&self) -> usize {
        self.pending_code.len()
    }

    /// Get active count
    pub fn active_count(&self) -> usize {
        self.active_code.len()
    }

    /// Save to file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Load from file or create new
    pub fn load_or_new(path: &str, sandbox_path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => {
                serde_json::from_str(&contents).unwrap_or_else(|e| {
                    warn!("Failed to parse codegen state: {}", e);
                    Self::new(sandbox_path)
                })
            }
            Err(_) => Self::new(sandbox_path),
        }
    }

    /// Format summary for logging
    pub fn format_summary(&self) -> String {
        format!(
            "{} active, {} pending, {} in history",
            self.active_count(),
            self.pending_count(),
            self.generator.history_count()
        )
    }
}

impl Default for CodeDeployer {
    fn default() -> Self {
        Self::new("/tmp/sovereign_sandbox")
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_literal() {
        let ctx = EvalContext::new(Regime::Ranging);
        let expr = Expression::lit(42.0);
        assert!((evaluate(&expr, &ctx) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_expression_variable() {
        let mut ctx = EvalContext::new(Regime::Ranging);
        ctx.set("test_var", 123.0);
        let expr = Expression::var("test_var");
        assert!((evaluate(&expr, &ctx) - 123.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_expression_compare() {
        let ctx = EvalContext::new(Regime::Ranging);

        let expr_lt = Expression::compare(
            Expression::lit(5.0),
            CompareOp::Lt,
            Expression::lit(10.0),
        );
        assert!(evaluate_bool(&expr_lt, &ctx));

        let expr_gt = Expression::compare(
            Expression::lit(5.0),
            CompareOp::Gt,
            Expression::lit(10.0),
        );
        assert!(!evaluate_bool(&expr_gt, &ctx));
    }

    #[test]
    fn test_expression_logic() {
        let ctx = EvalContext::new(Regime::Ranging);

        // true AND true = true
        let expr_and = Expression::and(Expression::True, Expression::True);
        assert!(evaluate_bool(&expr_and, &ctx));

        // true AND false = false
        let expr_and_f = Expression::and(Expression::True, Expression::False);
        assert!(!evaluate_bool(&expr_and_f, &ctx));

        // true OR false = true
        let expr_or = Expression::or(Expression::True, Expression::False);
        assert!(evaluate_bool(&expr_or, &ctx));

        // NOT true = false
        let expr_not = Expression::not(Expression::True);
        assert!(!evaluate_bool(&expr_not, &ctx));
    }

    #[test]
    fn test_expression_regime() {
        let ctx = EvalContext::new(Regime::Volatile);

        let expr_is = Expression::RegimeIs(Regime::Volatile);
        assert!(evaluate_bool(&expr_is, &ctx));

        let expr_is_not = Expression::RegimeIsNot(Regime::TrendingUp);
        assert!(evaluate_bool(&expr_is_not, &ctx));
    }

    #[test]
    fn test_expression_arithmetic() {
        let ctx = EvalContext::new(Regime::Ranging);

        let expr_add = Expression::add(Expression::lit(2.0), Expression::lit(3.0));
        assert!((evaluate(&expr_add, &ctx) - 5.0).abs() < f64::EPSILON);

        let expr_mul = Expression::mul(Expression::lit(4.0), Expression::lit(5.0));
        assert!((evaluate(&expr_mul, &ctx) - 20.0).abs() < f64::EPSILON);

        let expr_div = Expression::div(Expression::lit(10.0), Expression::lit(2.0));
        assert!((evaluate(&expr_div, &ctx) - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_expression_if_then_else() {
        let ctx = EvalContext::new(Regime::Ranging);

        let expr = Expression::if_then_else(
            Expression::True,
            Expression::lit(1.0),
            Expression::lit(0.0),
        );
        assert!((evaluate(&expr, &ctx) - 1.0).abs() < f64::EPSILON);

        let expr2 = Expression::if_then_else(
            Expression::False,
            Expression::lit(1.0),
            Expression::lit(0.0),
        );
        assert!((evaluate(&expr2, &ctx) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_expression_clamp() {
        let ctx = EvalContext::new(Regime::Ranging);

        let expr = Expression::clamp(
            Expression::lit(1.5),
            Expression::lit(0.0),
            Expression::lit(1.0),
        );
        assert!((evaluate(&expr, &ctx) - 1.0).abs() < f64::EPSILON);

        let expr2 = Expression::clamp(
            Expression::lit(-0.5),
            Expression::lit(0.0),
            Expression::lit(1.0),
        );
        assert!((evaluate(&expr2, &ctx) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sandbox_safety_check() {
        let sandbox = Sandbox::default();

        // Simple expression should pass
        let simple = Expression::var("test");
        assert!(sandbox.validate_safety(&simple).is_ok());

        // Create a deep expression
        let mut deep = Expression::lit(1.0);
        for _ in 0..25 {
            deep = Expression::add(deep.clone(), Expression::lit(1.0));
        }
        assert!(sandbox.validate_safety(&deep).is_err());
    }

    #[test]
    fn test_filter_generation() {
        let mut generator = CodeGenerator::default();

        let weakness = Weakness {
            weakness_type: WeaknessType::RegimeWeakness {
                regime: Regime::Volatile,
                win_rate: 0.35,
                trade_count: 50,
            },
            severity: 0.8,
            identified_at: Utc::now(),
            trades_analyzed: 50,
            suggested_action: "Avoid volatile regime".to_string(),
        };

        let code = generator.generate_signal_filter(&weakness);

        assert_eq!(code.code_type, CodeType::SignalFilter);
        assert!(!code.source_code.is_empty());
        assert!(code.description.contains("Volatile"));
    }

    #[test]
    fn test_adjuster_generation() {
        let mut generator = CodeGenerator::default();

        let insight = TradingInsight {
            insight_type: InsightType::ExitTooEarly,
            description: "Exiting positions too early".to_string(),
            evidence_count: 20,
            avg_improvement: 50.0,
            discovered_at: Utc::now(),
            symbol: None,
            regime: None,
        };

        let code = generator.generate_confidence_adjuster(&insight);

        assert_eq!(code.code_type, CodeType::ConfidenceAdjuster);
        assert!(!code.source_code.is_empty());
    }

    #[test]
    fn test_deployment_flow() {
        let mut deployer = CodeDeployer::default();

        // Generate and propose code
        let weakness = Weakness {
            weakness_type: WeaknessType::SRScoreWeakness {
                score_range: (-3, 0),
                win_rate: 0.4,
                trade_count: 30,
            },
            severity: 0.6,
            identified_at: Utc::now(),
            trades_analyzed: 30,
            suggested_action: "Require higher S/R".to_string(),
        };

        let code = deployer.generator_mut().generate_signal_filter(&weakness);
        let id = code.id;

        let result = deployer.propose_code(code);
        assert!(result.is_ok());

        // Deploy
        let deploy_result = deployer.deploy(id);
        assert!(deploy_result.is_ok());

        // Check it's active
        assert_eq!(deployer.active_count(), 1);
        assert!(deployer.get_active_filters().len() == 1);
    }

    #[test]
    fn test_rollback() {
        let mut deployer = CodeDeployer::default();

        let weakness = Weakness {
            weakness_type: WeaknessType::RegimeWeakness {
                regime: Regime::Ranging,
                win_rate: 0.45,
                trade_count: 40,
            },
            severity: 0.5,
            identified_at: Utc::now(),
            trades_analyzed: 40,
            suggested_action: "Caution in ranging".to_string(),
        };

        let code = deployer.generator_mut().generate_signal_filter(&weakness);
        let id = code.id;

        deployer.propose_code(code).unwrap();
        deployer.deploy(id).unwrap();

        assert_eq!(deployer.active_count(), 1);

        // Rollback
        deployer.rollback(id).unwrap();
        assert_eq!(deployer.active_count(), 0);
    }

    #[test]
    fn test_filter_execution() {
        let mut deployer = CodeDeployer::default();

        // Create filter that blocks volatile regime
        let weakness = Weakness {
            weakness_type: WeaknessType::RegimeWeakness {
                regime: Regime::Volatile,
                win_rate: 0.3,
                trade_count: 50,
            },
            severity: 0.9,
            identified_at: Utc::now(),
            trades_analyzed: 50,
            suggested_action: "Block volatile".to_string(),
        };

        let code = deployer.generator_mut().generate_signal_filter(&weakness);
        let id = code.id;

        deployer.propose_code(code).unwrap();
        deployer.deploy(id).unwrap();

        // Test execution - volatile should be blocked
        let ctx_volatile = EvalContext::new(Regime::Volatile);
        assert!(!deployer.execute_filters(&ctx_volatile));

        // Trending should pass
        let ctx_trending = EvalContext::new(Regime::TrendingUp);
        assert!(deployer.execute_filters(&ctx_trending));
    }

    #[test]
    fn test_adjuster_execution() {
        let mut deployer = CodeDeployer::default();

        let insight = TradingInsight {
            insight_type: InsightType::SizeTooLarge,
            description: "Sizes too large".to_string(),
            evidence_count: 15,
            avg_improvement: 30.0,
            discovered_at: Utc::now(),
            symbol: None,
            regime: None,
        };

        let code = deployer.generator_mut().generate_confidence_adjuster(&insight);
        let id = code.id;

        deployer.propose_code(code).unwrap();
        deployer.deploy(id).unwrap();

        // Execute adjuster
        let ctx = EvalContext::new(Regime::Ranging);
        let adjusted = deployer.execute_adjusters(0.7, &ctx);

        // Should be reduced (0.7 * 0.85 = 0.595)
        assert!(adjusted < 0.7);
        assert!(adjusted > 0.5);
    }

    #[test]
    fn test_code_performance_tracking() {
        let mut perf = CodePerformance::new();

        perf.record_execution(true, 50.0);
        perf.record_execution(true, -20.0);
        perf.record_execution(false, 0.0);

        assert_eq!(perf.times_executed, 3);
        assert_eq!(perf.trades_affected, 2);
        assert!((perf.actual_pnl_impact - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_expression_to_source() {
        let expr = Expression::compare(
            Expression::var("sr_score"),
            CompareOp::Gt,
            Expression::lit(-5.0),
        );

        let source = expr.to_source();
        assert!(source.contains("sr_score"));
        assert!(source.contains(">"));
    }

    #[test]
    fn test_backtest_results_scoring() {
        let good = BacktestResults {
            trades_simulated: 100,
            win_rate_change: 0.05,
            pnl_change: 100.0,
            sharpe_change: 0.2,
            max_dd_change: -0.02,
        };
        assert!(good.is_improvement());
        assert!(good.score() > 0.0);

        let bad = BacktestResults {
            trades_simulated: 100,
            win_rate_change: -0.05,
            pnl_change: -100.0,
            sharpe_change: -0.2,
            max_dd_change: 0.02,
        };
        assert!(!bad.is_improvement());
        assert!(bad.score() < 0.0);
    }
}
