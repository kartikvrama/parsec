"""Constants."""

SEED = 42  # Manual seed for all random functions.
EMPTY_LABEL = "empty"  # Dummy object label for an empty surface.
TEMPLATE_PH = "AGHRYTUK" # Placeholder to fill in prompt templates.
SUMMARY_PH = "BNHDORIT" # Placeholder to fill in LLM summaries.


class ENVIRONMENTS:
    """Constants for environment classes.

    Attributes:
        KITCHEN_CABINET: Kitchen cabinet environment. Ordinal spatial relations
            are defined with respect to the bottom/bottom-left shelf of each
            cabinet.
        BATHROOM_CABINET: Bathroom cabinet environment. Ordinal spatial
            relations are defined with respect to the bottom shelf.
        FRIDGE: Fridge environment. Ordinal spatial relations are defined with
            respect to the bottom shelf/drawer of each cabinet.
        BOOKSHELF: Bookshelf environment. Ordinal spatial relations are defined
            with respect to the bottom shelf.
        VANITY_DRESSER: Vanity dresser environment. Ordinal spatial relations
            are defined with respect to the bottom left corner of the
            environment.
        NOT_PLACED: Placeholder environment for objects that are not arranged.
        ENVIRONMENT_LIST: List of all environment types.
    """

    KITCHEN_CABINET = "kitchen"
    BATHROOM_CABINET = "bathroom"
    FRIDGE = "fridge"
    BOOKSHELF = "bookshelf"
    VANITY_DRESSER = "dresser"
    NOT_PLACED = "not_placed"

    # List of all environment types.
    ENVIRONMENT_LIST = [
        KITCHEN_CABINET,
        BATHROOM_CABINET,
        FRIDGE,
        BOOKSHELF,
        VANITY_DRESSER
    ]


# Mapping from HTML container labels to environment constants.
PAGE_TO_ENVIRONMENT = {
    "KitchenCabinet": ENVIRONMENTS.KITCHEN_CABINET,
    "BathroomCabinet": ENVIRONMENTS.BATHROOM_CABINET,
    "DresserDrawer": ENVIRONMENTS.VANITY_DRESSER,
    "Bookshelf": ENVIRONMENTS.BOOKSHELF,
    "Fridge": ENVIRONMENTS.FRIDGE,
}


class SURFACE_TYPES:
    """Constants for environment types.
    
    Attributes:
        DRAWER: Drawer surface type.
        CLOSED_SHELF: Closed shelf surface type.
        OPEN_SHELF: Open shelf surface type.
        DOOR_SHELF: Door shelf surface type (specifically for fridge).
        TABLE_SURFACE: Table surface type.
        NOT_PLACED: Placeholder surface type for objects that are not arranged.
        SURFACE_TYPE_LIST: List of all surface types.
    """

    DRAWER = "drawer"
    CLOSED_SHELF = "closed_shelf"
    OPEN_SHELF = "open_shelf"
    DOOR_SHELF = "door_shelf"
    TABLE_SURFACE = "table_surface"
    NOT_PLACED = "not_placed"

    # List of all non-empty surface types.
    SURFACE_TYPE_LIST = [
        DRAWER,
        CLOSED_SHELF,
        OPEN_SHELF,
        DOOR_SHELF,
        TABLE_SURFACE
    ]


class CATEGORIES:
    """Environment Semantic Categories
    """

    CAT_A = {
        ENVIRONMENTS.KITCHEN_CABINET: [2],
        ENVIRONMENTS.BATHROOM_CABINET: [3],
        ENVIRONMENTS.BOOKSHELF: [1, 2, 3],
    }
    CAT_B = {
        ENVIRONMENTS.KITCHEN_CABINET: [1, 3],
        ENVIRONMENTS.BATHROOM_CABINET: [2],
    }
    CAT_C = {
        ENVIRONMENTS.BATHROOM_CABINET: [1],
        ENVIRONMENTS.FRIDGE: [1, 2, 3],
        ENVIRONMENTS.VANITY_DRESSER: [1, 2, 3],
    }

    CATEGORY_LIST = [CAT_A, CAT_B, CAT_C]