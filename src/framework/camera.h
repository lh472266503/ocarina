//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "rhi/graphics_descriptions.h"
#include "math.h"

namespace ocarina {

class Camera {
public:
    Camera() = default;
    Camera(float fov, float aspect_ratio, float znear, float zfar)
        : fov_(fov), aspect_ratio_(aspect_ratio), znear_(znear), zfar_(zfar) {}
    virtual ~Camera() = default;
    float get_fov() const { return fov_; }
    void set_fov(float new_fov) {
        fov_ = new_fov;
        dirty_ = true;
    }
    float get_znear() const { return znear_; }
    void set_znear(float new_znear) {
        znear_ = new_znear;
        dirty_ = true;
    }
    float get_zfar() const { return zfar_; }
    void set_zfar(float new_zfar) {
        zfar_ = new_zfar;
        dirty_ = true;
    }

    float get_aspect_ratio() const { return aspect_ratio_; }
    void set_aspect_ratio(float new_aspect_ratio) {
        aspect_ratio_ = new_aspect_ratio;
        dirty_ = true;
    }

    void set_position(const math3d::Vector3D &position) {
        position_ = position;
        dirty_ = true;
    }

    void set_target(const math3d::Vector3D &target) {
        target_ = target;
        dirty_ = true;
    }

    math3d::Matrix4 get_view_matrix();
    math3d::Matrix4 get_projection_matrix();

    void update_matrices() {
        if (dirty_) {
            calculate_view_matrix();
            calculate_projection_matrix();
            dirty_ = false;
        }
    }

private:
    void calculate_view_matrix();
    void calculate_projection_matrix();
    float fov_ = 60.0f;
    float znear_ = 1.0f;
    float zfar_ = 100.0f;
    float aspect_ratio_ = 60.0f;// Default aspect ratio, unit: angle

    math3d::Vector3D position_;
    math3d::Vector3D target_ = {0, 0, 1};
    math3d::Vector3D up_ = {0, 1, 0};

    math3d::Matrix4 view_matrix_;// View matrix
    math3d::Matrix4 projection_matrix_;// Projection matrix

    bool dirty_ = true;// Indicates if the view or projection matrix needs to be recalculated
};

}// namespace ocarina