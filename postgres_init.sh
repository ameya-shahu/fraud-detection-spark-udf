#!/bin/bash

# Configuration
CONTAINER_NAME="postgres"
POSTGRES_PASSWORD="password"
POSTGRES_USER="postgres"
POSTGRES_DB="mydb"
CSV_FILE="./dataset/image_names.csv"
TABLE_NAME="mytable"

# Optional: Define columns for the CSV
TABLE_COLUMNS="id INT, name TEXT, age INT"

# Step 1: Remove old container if exists
docker rm -f $CONTAINER_NAME 2>/dev/null

# Step 2: Start fresh PostgreSQL container
echo "Starting PostgreSQL container..."
docker run -d \
  --name $CONTAINER_NAME \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_DB=$POSTGRES_DB \
  -p 5432:5432 \
  postgres:15

# Step 3: Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until docker exec $CONTAINER_NAME pg_isready -U $POSTGRES_USER &>/dev/null; do
  sleep 1
done
echo "PostgreSQL is ready!"

# Step 4: Create table (you can modify schema as needed)
echo "Creating table $TABLE_NAME..."
docker exec -i $CONTAINER_NAME psql -U $POSTGRES_USER -d $POSTGRES_DB <<EOF
DROP TABLE IF EXISTS $TABLE_NAME;
CREATE TABLE $TABLE_NAME ($TABLE_COLUMNS);
EOF

# Step 5: Copy CSV file into the container
echo "Copying CSV file to container..."
docker cp $CSV_FILE $CONTAINER_NAME:/tmp/$CSV_FILE

# Step 6: Load CSV data into the table
echo "Loading CSV data into $TABLE_NAME..."
docker exec -i $CONTAINER_NAME psql -U $POSTGRES_USER -d $POSTGRES_DB <<EOF
\COPY $TABLE_NAME FROM '/tmp/$CSV_FILE' WITH (FORMAT csv, HEADER);
EOF

echo "✅ Done. Data loaded into $TABLE_NAME!"
