"""Data structures for objects and environment surfaces."""

import copy
from typing import List, Dict, Optional, Union

from utils import constants


class ObjectEntity:
    """Data structure for objects.

    Attributes:
        object_id: Unique id of the object as defined in the dataset.
        name: Name of the object category.
    """

    object_id: str
    name: str

    def __init__(
        self,
        object_id: str,
        name: str,
    ) -> None:
        """Initializes an object."""

        self.object_id = object_id
        self.name = name

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.name

    def __eq__(self, __value: object) -> bool:
        """Check if two objects are equal by comparing object ids."""
        if isinstance(__value, ObjectEntity):
            return self.object_id == __value.object_id
        return False

    def __deepcopy__(self, memo: Dict[int, object]) -> "ObjectEntity":
        """Returns a deep copy of the object."""
        return ObjectEntity(self.object_id, self.name)


class SurfaceEntity:
    """Data structure for environment surfaces.

    Attributes:
        name: Name of the surface in json data.
        environment: Name of the environment the surface is in. Refer to
            constants.ENVIRONMENTS.
        surface_type: Type of surface. Refer to constants.SURFACE_TYPES.
        position: Position of the surface in the environment.
        text_label: Human readable label of the surface.
        objects_on_surface: List of objects placed on the surface. Refer to
            ObjectEntity class.
    """

    name: str
    environment: constants.ENVIRONMENTS
    surface_type: constants.SURFACE_TYPES
    position: Optional[List[int]]
    text_label: str
    objects_on_surface: List[ObjectEntity]

    def __init__(
        self,
        name: str,
        environment: constants.ENVIRONMENTS,
        surface_type: constants.SURFACE_TYPES,
        text_label: str,
        position: Optional[List[int]],
    ) -> None:
        """Initializes a surface object."""

        self.name = name
        self.environment = environment
        self.surface_type = surface_type
        self.position = position
        self.text_label = text_label
        self.objects_on_surface = []

    def __str__(self) -> str:
        """Return a string representation of the surface."""
        return f"{self.name}-{self.position}"

    def __eq__(self, __value: object) -> bool:
        """Check if two surfaces are semantically equal."""
        if isinstance(__value, SurfaceEntity):
            return (
                self.environment == __value.environment
                and self.surface_type == __value.surface_type
                and self.position == __value.position
            )
        return False

    def __deepcopy__(self, memo: Dict[int, object]) -> "SurfaceEntity":
        """Returns a deep copy of the surface entity."""
        new_surface = SurfaceEntity(
            name=self.name, environment=self.environment,
            surface_type=self.surface_type, text_label=self.text_label, position=self.position
        )
        if not self.objects_on_surface:
            return new_surface
        new_objects = [
            copy.deepcopy(object) for object in self.objects_on_surface
        ]
        new_surface.add_objects(new_objects)
        return new_surface

    def to_dict(self) -> Dict[str, Union[List[Dict[str, str]], str]]:
        """Returns a dictionary representation of the surface."""
        return {
            "surface_name": self.name,
            "environment": self.environment,
            "surface_type": self.surface_type,
            "position": self.position,
            "text_label": self.text_label,
            "objects_on_surface": [{
                "object_id": object_entity.object_id,
                "object_name": object_entity.name
                } for object_entity in self.objects_on_surface
            ],
        }

    def add_objects(self, objects_to_add: List[ObjectEntity]) -> None:
        """Adds object to the surface.

        Args:
            objects_to_add: List of objects to be added to the surface.
        """
        if not isinstance(objects_to_add, list):
            raise ValueError(
                "Input should be a list of ObjectEntity objects."
            )
        self.objects_on_surface.extend(objects_to_add)
