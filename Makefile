.PHONY: image

image:
	podman build --tag ubuntu:bdann -f ./podman/CUDA11.Dockerfile
