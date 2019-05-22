/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Cordova Estrada <christianc@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GIS_HPP
#define GIS_HPP

#include "cudf.h"

namespace cudf {

/** 
 * @brief Determine whether or not coordinates (query points) are completely inside a static polygon
 * 
 * Polygon is considered as a set of coordinates (latitudes and longitudes)
 * where the first and last coordinates must have the same value (closed)
 * 
 * It works for clockwise and counter-clockwise polygons
 * 
 * Polygon must not have holes
 * 
 * If a query point is colinear with two contiguous polygon's coordinates
 * then this query point isn't completely inside
 * 
 * Both polygon_latitudes and polygon_longitudes must have equal size
 * 
 * Both point_latitudes and point_longitudes must have equal size
 * 
 * All input params must have equal datatypes (for numeric operations)
 *
 * @param[in] polygon_latitudes: column with latitudes of a polygon
 * @param[in] polygon_longitudes: column with longitudes of a polygon
 * @param[in] point_latitudes: column with latitudes of query points
 * @param[in] point_longitudes: column with longitudes of query points
 *
 * @returns gdf_column of type GDF_BOOL8 indicating whether the i-th query point is inside (true) or not (false)
 */
gdf_column point_in_polygon(gdf_column const & polygon_latitudes,
                            gdf_column const & polygon_longitudes,
	                        gdf_column const & point_latitudes,
                            gdf_column const & point_longitudes);

}  // namespace cudf

#endif  // GIS_H