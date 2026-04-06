"""Initial Fitbit Sleep Score schema."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


sleep_session_type = sa.Enum(
    "stages",
    "classic",
    name="sleep_session_type",
    native_enum=False,
    validate_strings=True,
)
sleep_stage_type = sa.Enum(
    "wake",
    "light",
    "deep",
    "rem",
    "asleep",
    "restless",
    name="sleep_stage_type",
    native_enum=False,
    validate_strings=True,
)
recommendation_category = sa.Enum(
    "sleep_schedule",
    "sleep_hygiene",
    "recovery",
    "stress",
    "activity",
    "environment",
    "clinical_follow_up",
    name="recommendation_category",
    native_enum=False,
    validate_strings=True,
)


def upgrade() -> None:
    """Create the initial application schema."""

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("fitbit_user_id", sa.String(length=64), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("birth_date", sa.Date(), nullable=True),
        sa.Column("sex", sa.String(length=32), nullable=True),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="1", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.UniqueConstraint("fitbit_user_id", name="uq_users_fitbit_user_id"),
    )
    op.create_index("ix_users_timezone", "users", ["timezone"], unique=False)

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("framework", sa.String(length=50), server_default="PyTorch", nullable=False),
        sa.Column("artifact_uri", sa.String(length=500), nullable=True),
        sa.Column("training_metrics", sa.JSON(), nullable=True),
        sa.Column("feature_schema", sa.JSON(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_versions")),
        sa.UniqueConstraint("version", name="uq_model_versions_version"),
    )
    op.create_index("ix_model_versions_is_active", "model_versions", ["is_active"], unique=False)
    op.create_index("ix_model_versions_model_name", "model_versions", ["model_name"], unique=False)

    op.create_table(
        "sleep_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("fitbit_log_id", sa.Integer(), nullable=False),
        sa.Column("session_date", sa.Date(), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_minutes", sa.Integer(), nullable=True),
        sa.Column("time_in_bed", sa.Integer(), nullable=True),
        sa.Column("minutes_asleep", sa.Integer(), nullable=True),
        sa.Column("minutes_awake", sa.Integer(), nullable=True),
        sa.Column("minutes_after_wakeup", sa.Integer(), nullable=True),
        sa.Column("minutes_to_fall_asleep", sa.Integer(), nullable=True),
        sa.Column("awakenings_count", sa.Integer(), nullable=True),
        sa.Column("efficiency", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("is_main_sleep", sa.Boolean(), server_default="1", nullable=False),
        sa.Column("session_type", sleep_session_type, nullable=True),
        sa.Column("source_payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_sleep_sessions_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sleep_sessions")),
        sa.UniqueConstraint("fitbit_log_id", name="uq_sleep_sessions_fitbit_log_id"),
    )
    op.create_index("ix_sleep_sessions_main_sleep", "sleep_sessions", ["user_id", "is_main_sleep"], unique=False)
    op.create_index("ix_sleep_sessions_user_date", "sleep_sessions", ["user_id", "session_date"], unique=False)
    op.create_index("ix_sleep_sessions_user_start_time", "sleep_sessions", ["user_id", "start_time"], unique=False)

    op.create_table(
        "sleep_stages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("sleep_session_id", sa.Integer(), nullable=False),
        sa.Column("stage_type", sleep_stage_type, nullable=False),
        sa.Column("sequence_index", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("seconds", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["sleep_session_id"],
            ["sleep_sessions.id"],
            name=op.f("fk_sleep_stages_sleep_session_id_sleep_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sleep_stages")),
    )
    op.create_index("ix_sleep_stages_session_sequence", "sleep_stages", ["sleep_session_id", "sequence_index"], unique=False)
    op.create_index("ix_sleep_stages_session_stage_type", "sleep_stages", ["sleep_session_id", "stage_type"], unique=False)
    op.create_index("ix_sleep_stages_started_at", "sleep_stages", ["started_at"], unique=False)

    op.create_table(
        "sleep_scores",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("sleep_session_id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=True),
        sa.Column("score_date", sa.Date(), nullable=False),
        sa.Column("overall_score", sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("sleep_efficiency_score", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("duration_score", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("continuity_score", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("recovery_score", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("feature_snapshot", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], name=op.f("fk_sleep_scores_model_version_id_model_versions"), ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["sleep_session_id"], ["sleep_sessions.id"], name=op.f("fk_sleep_scores_sleep_session_id_sleep_sessions"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_sleep_scores_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sleep_scores")),
        sa.UniqueConstraint("sleep_session_id", name="uq_sleep_scores_sleep_session_id"),
    )
    op.create_index("ix_sleep_scores_model_version", "sleep_scores", ["model_version_id"], unique=False)
    op.create_index("ix_sleep_scores_overall_score", "sleep_scores", ["overall_score"], unique=False)
    op.create_index("ix_sleep_scores_user_score_date", "sleep_scores", ["user_id", "score_date"], unique=False)

    op.create_table(
        "sleep_recommendations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("sleep_score_id", sa.Integer(), nullable=True),
        sa.Column("recommendation_date", sa.Date(), nullable=False),
        sa.Column("category", recommendation_category, nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("priority", sa.Integer(), server_default="1", nullable=False),
        sa.Column("action_items", sa.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="1", nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["sleep_score_id"], ["sleep_scores.id"], name=op.f("fk_sleep_recommendations_sleep_score_id_sleep_scores"), ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_sleep_recommendations_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sleep_recommendations")),
    )
    op.create_index("ix_sleep_recommendations_category", "sleep_recommendations", ["category"], unique=False)
    op.create_index("ix_sleep_recommendations_is_active", "sleep_recommendations", ["is_active"], unique=False)
    op.create_index("ix_sleep_recommendations_score_id", "sleep_recommendations", ["sleep_score_id"], unique=False)
    op.create_index("ix_sleep_recommendations_user_date", "sleep_recommendations", ["user_id", "recommendation_date"], unique=False)


def downgrade() -> None:
    """Drop the initial application schema."""

    op.drop_index("ix_sleep_recommendations_user_date", table_name="sleep_recommendations")
    op.drop_index("ix_sleep_recommendations_score_id", table_name="sleep_recommendations")
    op.drop_index("ix_sleep_recommendations_is_active", table_name="sleep_recommendations")
    op.drop_index("ix_sleep_recommendations_category", table_name="sleep_recommendations")
    op.drop_table("sleep_recommendations")

    op.drop_index("ix_sleep_scores_user_score_date", table_name="sleep_scores")
    op.drop_index("ix_sleep_scores_overall_score", table_name="sleep_scores")
    op.drop_index("ix_sleep_scores_model_version", table_name="sleep_scores")
    op.drop_table("sleep_scores")

    op.drop_index("ix_sleep_stages_started_at", table_name="sleep_stages")
    op.drop_index("ix_sleep_stages_session_stage_type", table_name="sleep_stages")
    op.drop_index("ix_sleep_stages_session_sequence", table_name="sleep_stages")
    op.drop_table("sleep_stages")

    op.drop_index("ix_sleep_sessions_user_start_time", table_name="sleep_sessions")
    op.drop_index("ix_sleep_sessions_user_date", table_name="sleep_sessions")
    op.drop_index("ix_sleep_sessions_main_sleep", table_name="sleep_sessions")
    op.drop_table("sleep_sessions")

    op.drop_index("ix_model_versions_model_name", table_name="model_versions")
    op.drop_index("ix_model_versions_is_active", table_name="model_versions")
    op.drop_table("model_versions")

    op.drop_index("ix_users_timezone", table_name="users")
    op.drop_table("users")
