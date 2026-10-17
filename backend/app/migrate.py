from app.postgres_store import DatabaseUnavailableError, migrate_database


def main() -> None:
    try:
        migrate_database()
    except DatabaseUnavailableError as error:
        raise SystemExit(f"Database migration failed: {error}") from error
    print("TalentLens AI PostgreSQL migrations applied.")


if __name__ == "__main__":
    main()
