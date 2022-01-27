.PHONY: pimage pclean prun pcleanall

TAG=ubuntu:bdann
DOCKERFILE=./podman/CUDA11.Dockerfile
CONTAINERNAME=bdann1

pimage:
	podman build --tag $(TAG) -f $(DOCKERFILE)
pclean:
	podman rmi $(TAG)
prun:
	podman run -d -it --name=$(CONTAINERNAME) --shm-size=4g -v .:/bdann --security-opt=label=disable $(TAG)
pcleanall:
	podman system prune --all --force && podman rmi --all
