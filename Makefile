# Docker configuration
DOCKER_IMAGE_NAME = rabi-oscillation-sonification
DOCKER_CONTAINER_NAME = rabi-oscillation-sonification
PORT = 8501

# Build the Docker image
build:
	docker build -t $(DOCKER_IMAGE_NAME) .

# Run the container
run:
	docker run --name $(DOCKER_CONTAINER_NAME) -p $(PORT):$(PORT) -d $(DOCKER_IMAGE_NAME)

# View container logs
logs:
	docker logs -f $(DOCKER_CONTAINER_NAME)

# Stop and remove container
stop:
	docker stop $(DOCKER_CONTAINER_NAME) || true
	docker rm $(DOCKER_CONTAINER_NAME) || true

# Remove image
rmi:
	docker rmi $(DOCKER_IMAGE_NAME) || true

# Clean everything (container and image)
clean: stop rmi

# Build and run in one command
up: build run

# Default help command
help:
	@echo "Available commands:"
	@echo "  make build      - Build the Docker image"
	@echo "  make run        - Run the Docker container"
	@echo "  make up         - Build and run in one command"
	@echo "  make logs       - View container logs"
	@echo "  make stop       - Stop and remove the container"
	@echo "  make rmi        - Remove the Docker image"
	@echo "  make clean      - Remove container and image"

.PHONY: build run logs stop rmi clean up help
